from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class ImageForceAttentionEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # cross-attention parameters
            force_projection_dim: int=512,
            attention_num_heads: int=8,
            attention_dropout: float=0.1
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        
        New parameters:
        - force_projection_dim: Dimension to project force features to match image features
        - attention_num_heads: Number of attention heads for cross-attention
        - attention_dropout: Dropout rate for attention layers
        """
        super().__init__()

        rgb_keys = list()
        low_dim_no_force_keys = list()
        force_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        # print('encoder, obs_shape_meta', obs_shape_meta)
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                # Categorize low-dim keys as force or non-force based on key name
                if 'force' in key.lower() or 'wrench' in key.lower():
                    force_keys.append(key)
                else:
                    low_dim_no_force_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        rgb_keys = sorted(rgb_keys)
        low_dim_no_force_keys = sorted(low_dim_no_force_keys)
        force_keys = sorted(force_keys)
        
        # Calculate total force dimension
        total_force_dim = sum(key_shape_map[key][0] for key in force_keys) if force_keys else 0
        total_lowdim_dim = sum(key_shape_map[key][0] for key in low_dim_no_force_keys) if low_dim_no_force_keys else 0
        
        # Initialize cross-attention components
        self.force_projection_dim = force_projection_dim
        self.has_force_data = total_force_dim > 0
        self.has_lowdim_data = total_lowdim_dim > 0
        
        if self.has_force_data:
            # Linear projection to map force features to match image feature dimension
            self.force_projection = nn.Linear(total_force_dim, force_projection_dim)
            
            # Cross-attention layer (force attends to image)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=force_projection_dim,
                num_heads=attention_num_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            
            # Layer normalization for attention output
            self.attention_norm = nn.LayerNorm(force_projection_dim)
            
            # Final projection for joint embedding
            self.joint_projection = nn.Sequential(
                nn.Linear(force_projection_dim * 2, force_projection_dim),  # Concatenated image+force features
                nn.ReLU(),
                nn.Dropout(attention_dropout),
                nn.Linear(force_projection_dim, force_projection_dim)
            )

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_no_force_keys = low_dim_no_force_keys
        self.force_keys = force_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        
        # Step 1: Process RGB input to get image features
        image_features = None
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            image_features = feature.reshape(batch_size,-1)
        else:
            # run each rgb obs to independent models
            img_feats = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                img_feats.append(feature)
            if img_feats:
                image_features = torch.cat(img_feats, dim=-1)
        
        # Step 2: Process force data
        force_features = None
        if self.has_force_data and self.force_keys:
            force_data = []
            for key in self.force_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                # print('force encoder lowdim shape:', key, data.shape)
                assert data.shape[1:] == self.key_shape_map[key]
                force_data.append(data)
            
            if force_data:
                # Concatenate all force features
                concatenated_force = torch.cat(force_data, dim=-1)  # [B, total_force_dim]
                
                # Project force features to match image feature dimension
                force_features = self.force_projection(concatenated_force)  # [B, force_projection_dim]
        
        # Step 3: Cross-attention between image and force features
        joint_features = None
        if image_features is not None and force_features is not None:
            # Prepare features for cross-attention
            # Image features as key/value, force features as query
            force_query = force_features.unsqueeze(1)  # [B, 1, force_projection_dim]
            
            # Check that image features match force projection dimension
            if image_features.shape[-1] != self.force_projection_dim:
                raise RuntimeError(
                    f"Image features dimension ({image_features.shape[-1]}) does not match "
                    f"force projection dimension ({self.force_projection_dim}). "
                    f"Please ensure the RGB model output dimension matches the force_projection_dim parameter, "
                    f"or use dynamic dimension detection in the constructor."
                )
            
            image_key_value = image_features.unsqueeze(1)  # [B, 1, force_projection_dim]
            
            # Cross-attention: force attends to image
            attended_features, attention_weights = self.cross_attention(
                query=force_query,      # [B, 1, force_projection_dim]
                key=image_key_value,    # [B, 1, force_projection_dim]
                value=image_key_value   # [B, 1, force_projection_dim]
            )
            attended_features = attended_features.squeeze(1)  # [B, force_projection_dim]
            
            # Apply layer normalization
            attended_features = self.attention_norm(attended_features)
            
            # Create joint embedding by concatenating original force and attended features
            joint_input = torch.cat([force_features, attended_features], dim=-1)
            joint_features = self.joint_projection(joint_input)  # [B, force_projection_dim]
            
        elif image_features is not None:
            # Only image features available
            joint_features = image_features
        elif force_features is not None:
            # Only force features available (unlikely case)
            joint_features = force_features
        
        # Step 4: Process pose data and concatenate at the end
        final_features = []
        if joint_features is not None:
            final_features.append(joint_features)
        
        # Add pose features at the end
        for key in self.low_dim_no_force_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # print('pose encoder lowdim shape:', key, data.shape)
            assert data.shape[1:] == self.key_shape_map[key]
            final_features.append(data)

        result = torch.cat(final_features, dim=-1)
        
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
