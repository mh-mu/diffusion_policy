#!/usr/bin/env python3

"""
Test script for the ImageForceAttentionEncoder to verify it works correctly
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from diffusion_policy.model.vision.image_force_attention_encoder import ImageForceAttentionEncoder
from diffusion_policy.model.vision.model_getter import get_resnet


def create_test_shape_meta():
    """Create test shape_meta for different scenarios"""
    return {
        'obs': {
            'camera_0': {
                'shape': [3, 480, 640],
                'type': 'rgb'
            },
            'force': {
                'shape': [6],
                'type': 'low_dim'
            },
            'agent_pos': {
                'shape': [3],
                'type': 'low_dim'
            }
        }
    }


def create_test_obs_dict(batch_size=2):
    """Create test observation dictionary"""
    return {
        'camera_0': torch.randn(batch_size, 3, 480, 640),
        'force': torch.randn(batch_size, 6),
        'agent_pos': torch.randn(batch_size, 3)
    }


def test_encoder_initialization():
    """Test basic encoder initialization"""
    print("Testing encoder initialization...")
    
    shape_meta = create_test_shape_meta()
    rgb_model = get_resnet('resnet18')
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        force_projection_dim=512,  # Match ResNet18 output
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    print(f"✓ Encoder initialized successfully")
    print(f"  RGB keys: {encoder.rgb_keys}")
    print(f"  Force keys: {encoder.force_keys}")
    print(f"  Low-dim (non-force) keys: {encoder.low_dim_no_force_keys}")
    print(f"  Force projection dim: {encoder.force_projection_dim}")
    print(f"  Has force data: {encoder.has_force_data}")


def test_encoder_forward_pass():
    """Test forward pass with matched dimensions"""
    print("\nTesting forward pass with matched dimensions...")
    
    shape_meta = create_test_shape_meta()
    rgb_model = get_resnet('resnet18')  # 512 output dim
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        force_projection_dim=512,  # Match ResNet18 output
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    obs_dict = create_test_obs_dict(batch_size=4)
    
    # Test forward pass
    with torch.no_grad():
        output = encoder(obs_dict)
    
    print(f"✓ Forward pass successful")
    print(f"  Input batch size: {obs_dict['camera_0'].shape[0]}")
    print(f"  Output shape: {output.shape}")
    
    # Check output shape
    expected_output_dim = 512 + 3  # force_projection_dim + agent_pos_dim
    assert output.shape == (4, expected_output_dim), f"Expected shape (4, {expected_output_dim}), got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")


def test_dimension_mismatch_error():
    """Test that dimension mismatch raises appropriate error"""
    print("\nTesting dimension mismatch error...")
    
    shape_meta = create_test_shape_meta()
    rgb_model = get_resnet('resnet50')  # 2048 output dim
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        force_projection_dim=512,  # Mismatch: ResNet50 outputs 2048
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    obs_dict = create_test_obs_dict(batch_size=2)
    
    # This should raise a RuntimeError
    try:
        with torch.no_grad():
            output = encoder(obs_dict)
        assert False, "Expected RuntimeError but none was raised"
    except RuntimeError as e:
        expected_msg = "Image features dimension (2048) does not match force projection dimension (512)"
        assert expected_msg in str(e), f"Error message doesn't match expected pattern: {str(e)}"
        print(f"✓ Dimension mismatch error correctly raised: {str(e)}")


def test_encoder_without_force():
    """Test encoder when no force data is present"""
    print("\nTesting encoder without force data...")
    
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, 84, 84],
                'type': 'rgb'
            },
            'agent_pos': {
                'shape': [7],
                'type': 'low_dim'
            }
        }
    }
    
    rgb_model = get_resnet('resnet18')
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        force_projection_dim=512,
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    obs_dict = {
        'camera_0': torch.randn(2, 3, 84, 84),
        'agent_pos': torch.randn(2, 7)
    }
    
    assert not encoder.has_force_data, "Should not have force data"
    
    with torch.no_grad():
        output = encoder(obs_dict)
    
    # Should output RGB features + agent_pos features
    expected_output_dim = 512 + 7  # RGB features + agent_pos
    assert output.shape == (2, expected_output_dim), f"Expected shape (2, {expected_output_dim}), got {output.shape}"
    print(f"✓ No-force case works correctly: {output.shape}")


def test_encoder_without_rgb():
    """Test encoder when no RGB data is present"""
    print("\nTesting encoder without RGB data...")
    
    shape_meta = {
        'obs': {
            'force': {
                'shape': [6],
                'type': 'low_dim'
            },
            'agent_pos': {
                'shape': [7],
                'type': 'low_dim'
            }
        }
    }
    
    # Empty RGB model since no RGB data
    rgb_model = get_resnet('resnet18')
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        force_projection_dim=512,
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    obs_dict = {
        'force': torch.randn(2, 6),
        'agent_pos': torch.randn(2, 7)
    }
    
    assert encoder.has_force_data, "Should have force data"
    assert len(encoder.rgb_keys) == 0, "Should not have RGB keys"
    
    with torch.no_grad():
        output = encoder(obs_dict)
    
    # Should output force features + agent_pos features
    expected_output_dim = 512 + 7  # force_projection_dim + agent_pos
    assert output.shape == (2, expected_output_dim), f"Expected shape (2, {expected_output_dim}), got {output.shape}"
    print(f"✓ No-RGB case works correctly: {output.shape}")


def test_encoder_shared_rgb_model():
    """Test encoder with shared RGB model"""
    print("\nTesting encoder with shared RGB model...")
    
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, 84, 84],
                'type': 'rgb'
            },
            'camera_1': {
                'shape': [3, 84, 84],
                'type': 'rgb'
            },
            'force': {
                'shape': [6],
                'type': 'low_dim'
            },
            'agent_pos': {
                'shape': [7],
                'type': 'low_dim'
            }
        }
    }
    
    rgb_model = get_resnet('resnet18')
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        share_rgb_model=True,  # Share model between cameras
        force_projection_dim=1024,  # 512 * 2 cameras
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    obs_dict = {
        'camera_0': torch.randn(2, 3, 84, 84),
        'camera_1': torch.randn(2, 3, 84, 84),
        'force': torch.randn(2, 6),
        'agent_pos': torch.randn(2, 7)
    }
    
    with torch.no_grad():
        output = encoder(obs_dict)
    
    # Should output joint features + agent_pos features
    expected_output_dim = 1024 + 7  # force_projection_dim + agent_pos
    assert output.shape == (2, expected_output_dim), f"Expected shape (2, {expected_output_dim}), got {output.shape}"
    print(f"✓ Shared RGB model works correctly: {output.shape}")


def test_output_shape_method():
    """Test the output_shape method"""
    print("\nTesting output_shape method...")
    
    shape_meta = create_test_shape_meta()
    rgb_model = get_resnet('resnet18')
    
    encoder = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        force_projection_dim=512,
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    # Test output_shape method
    output_shape = encoder.output_shape()
    print(f"✓ Output shape method works: {output_shape}")
    
    # Verify it matches actual output
    obs_dict = create_test_obs_dict(batch_size=1)
    with torch.no_grad():
        actual_output = encoder(obs_dict)
    
    assert output_shape == actual_output.shape[1:], f"Output shape method mismatch: {output_shape} vs {actual_output.shape[1:]}"
    print(f"✓ Output shape method matches actual output")


def test_different_resnet_models():
    """Test with different ResNet models"""
    print("\nTesting different ResNet models...")
    
    shape_meta = create_test_shape_meta()
    
    # Test ResNet18 (512 features)
    resnet18 = get_resnet('resnet18')
    encoder18 = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=resnet18,
        force_projection_dim=512,
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    # Test ResNet50 (2048 features)
    resnet50 = get_resnet('resnet50')
    encoder50 = ImageForceAttentionEncoder(
        shape_meta=shape_meta,
        rgb_model=resnet50,
        force_projection_dim=2048,
        attention_num_heads=8,
        attention_dropout=0.1
    )
    
    obs_dict = create_test_obs_dict(batch_size=2)
    
    with torch.no_grad():
        output18 = encoder18(obs_dict)
        output50 = encoder50(obs_dict)
    
    print(f"✓ ResNet18 encoder output: {output18.shape}")
    print(f"✓ ResNet50 encoder output: {output50.shape}")
    
    # Both should have same final dimension: force_projection_dim + agent_pos
    assert output18.shape == (2, 512 + 3)
    assert output50.shape == (2, 2048 + 3)


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("TESTING IMAGE FORCE ATTENTION ENCODER")
    print("=" * 60)
    
    try:
        test_encoder_initialization()
        test_encoder_forward_pass()
        test_dimension_mismatch_error()
        test_encoder_without_force()
        test_encoder_without_rgb()
        test_encoder_shared_rgb_model()
        test_output_shape_method()
        test_different_resnet_models()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()