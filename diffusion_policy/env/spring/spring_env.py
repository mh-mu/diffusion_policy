import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class SpringEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
    
    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward based on spring force
        # Success when spring force reaches target threshold
        target_force = 5.0  # 5N target force
        current_force = getattr(self, 'current_force_magnitude', 0)
        reward = min(1.0, current_force / target_force)
        done = current_force == target_force

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        
        # Calculate spring information and forces
        spring_info = {}
        total_force_magnitude = 0
        spring_force_vector = Vec2d(0, 0)
        
        if hasattr(self, 'spring_anchor') and self.spring_anchor is not None:
            # Calculate spring properties - positions are already Vec2d objects
            spring_vector = self.spring_anchor.position - self.block.position
            current_length = spring_vector.length
            rest_length = self.spring_constraint.rest_length if hasattr(self, 'spring_constraint') else 100
            displacement = current_length - rest_length
            spring_stiffness = self.spring_constraint.stiffness if hasattr(self, 'spring_constraint') else 5
            
            # Calculate spring force vector with direction (Hooke's law: F = -k * displacement_vector)
            if current_length > 0:
                # Unit vector from tee to anchor
                spring_unit_vector = spring_vector.normalized()
                # Force magnitude (positive for extension, negative for compression)
                force_magnitude = displacement * spring_stiffness
                # Force vector on the tee (points toward anchor when extended, away when compressed)
                spring_force_vector = spring_unit_vector * force_magnitude
                total_force_magnitude = abs(force_magnitude)
            else:
                spring_force_vector = Vec2d(0, 0)
                total_force_magnitude = 0
            
            spring_info.update({
                'spring_length': current_length,
                'spring_rest_length': rest_length,
                'spring_displacement': displacement,
                'spring_force_magnitude': total_force_magnitude,
                'spring_force_vector': np.array(spring_force_vector),
                'spring_anchor_pos': np.array(self.spring_anchor.position),
                'spring_compressed': displacement < 0,
                'spring_extended': displacement > 0,
                'force_direction': 'toward_anchor' if displacement > 0 else 'away_from_anchor' if displacement < 0 else 'none'
            })
        
        # Store force info for rendering
        self.current_force_magnitude = total_force_magnitude
        self.current_force_vector = spring_force_vector
        
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'n_contacts': n_contact_points_per_step,
            'total_force_on_tee': total_force_magnitude,
            **spring_info  # Add spring information
        }
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw spring connection between rectangle and anchor circle
        if hasattr(self, 'spring_anchor') and self.spring_anchor is not None:
            # Calculate spring properties for visualization
            spring_start = pymunk.pygame_util.to_pygame(self.block.position, canvas)
            spring_end = pymunk.pygame_util.to_pygame(self.spring_anchor.position, canvas)
            
            # Calculate current spring length vs rest length
            current_length = (self.block.position - self.spring_anchor.position).length
            rest_length = 100  # Should match the rest_length in spring constraint
            
            # Color based on compression/extension
            if current_length < rest_length * 0.9:  # Compressed
                spring_color = pygame.Color('Red')
                thickness = 4
            elif current_length > rest_length * 1.1:  # Extended
                spring_color = pygame.Color('Blue') 
                thickness = 4
            else:  # Near rest length
                spring_color = pygame.Color('Green')
                thickness = 2
            
            # Draw spring line
            pygame.draw.line(canvas, spring_color, spring_start, spring_end, thickness)

        # Draw agent and block.
        self.space.debug_draw(draw_options)
        
        # Display force information on the tee block
        if hasattr(self, 'current_force_magnitude') and hasattr(self, 'current_force_vector'):
            # Get tee position in pygame coordinates
            tee_pos_pygame = pymunk.pygame_util.to_pygame(self.block.position, canvas)
            
            # Display force magnitude and direction as text
            font = pygame.font.Font(None, 20)
            force_text = f"Force: {self.current_force_magnitude:.1f}N"
            text_surface = font.render(force_text, True, pygame.Color('Black'))
            # Position text above the tee
            text_pos = (tee_pos_pygame[0] - text_surface.get_width()//2, tee_pos_pygame[1] - 50)
            canvas.blit(text_surface, text_pos)
            
            # Display force direction
            if self.current_force_magnitude > 0.1:
                if hasattr(self, 'spring_anchor') and self.spring_anchor is not None:
                    current_length = (self.spring_anchor.position - self.block.position).length
                    rest_length = 100
                    if current_length > rest_length:
                        direction_text = "→ Pulling toward anchor"
                        color = pygame.Color('Blue')
                    elif current_length < rest_length:
                        direction_text = "← Pushing away from anchor"
                        color = pygame.Color('Red')
                    else:
                        direction_text = "○ At rest"
                        color = pygame.Color('Green')
                    
                    direction_surface = font.render(direction_text, True, color)
                    direction_pos = (tee_pos_pygame[0] - direction_surface.get_width()//2, tee_pos_pygame[1] - 30)
                    canvas.blit(direction_surface, direction_pos)
            
            # Draw force vector arrow if force is significant
            if self.current_force_magnitude > 0.5:  # Only show if force > 0.5N
                # Scale the force vector for visualization
                force_scale = min(60, max(20, self.current_force_magnitude * 2))  # Scale between 20-60 pixels
                
                # Calculate force vector end position
                force_vector_normalized = self.current_force_vector.normalized()
                force_end_world = self.block.position + (force_vector_normalized * force_scale)
                force_end_pygame = pymunk.pygame_util.to_pygame(force_end_world, canvas)
                
                # Choose arrow color based on force type
                if self.current_force_magnitude > 10:
                    arrow_color = pygame.Color('DarkRed')
                    arrow_width = 4
                elif self.current_force_magnitude > 5:
                    arrow_color = pygame.Color('Red')
                    arrow_width = 3
                else:
                    arrow_color = pygame.Color('Orange')
                    arrow_width = 2
                
                # Draw force vector arrow
                pygame.draw.line(canvas, arrow_color, tee_pos_pygame, force_end_pygame, arrow_width)
                
                # Draw arrowhead
                import math
                arrow_length = 8
                arrow_angle = math.atan2(force_end_pygame[1] - tee_pos_pygame[1], 
                                       force_end_pygame[0] - tee_pos_pygame[0])
                
                # Calculate arrowhead points
                arrow_point1 = (
                    force_end_pygame[0] - arrow_length * math.cos(arrow_angle - math.pi/6),
                    force_end_pygame[1] - arrow_length * math.sin(arrow_angle - math.pi/6)
                )
                arrow_point2 = (
                    force_end_pygame[0] - arrow_length * math.cos(arrow_angle + math.pi/6),
                    force_end_pygame[1] - arrow_length * math.sin(arrow_angle + math.pi/6)
                )
                
                # Draw filled arrowhead
                pygame.draw.polygon(canvas, arrow_color, 
                                  [force_end_pygame, arrow_point1, arrow_point2])

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Initialize force tracking variables
        self.current_force_magnitude = 0
        self.current_force_vector = Vec2d(0, 0)
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent and dynamic rectangle with spring
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_dynamic_rectangle((256, 300), 40, 20)  # Dynamic rectangle instead of tee
        
        # Add spring anchor circle (this will be connected to the rectangle via spring)
        self.spring_anchor = self.add_dynamic_circle((356, 300), 10)  # 100 pixels to the right of rectangle
        
        # Create spring constraint between rectangle and anchor circle
        self.spring_constraint = pymunk.DampedSpring(
            self.block,           # First body (rectangle)
            self.spring_anchor,   # Second body (anchor circle)
            (0, 0),              # Anchor point on rectangle (center)
            (0, 0),              # Anchor point on circle (center)
            100,                 # Rest length (natural spring length)
            5,                 # Stiffness (spring constant k)
            20                   # Damping coefficient
        )
        self.space.add(self.spring_constraint)
        
        # Optional: Add a visual spring constraint (for better visualization)
        # This doesn't add physics, just helps with debugging
        self.spring_constraint.collide_bodies = False  # Don't collide spring bodies with each other

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        # Target force for success condition (5N as requested)
        self.target_force = 5.0

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_dynamic_circle(self, position, radius, mass=1):
        """Add a dynamic circle that can move and respond to forces"""
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.STATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('Orange')  # Different color to distinguish from agent
        shape.friction = 0.7
        self.space.add(body, shape)
        return body

    def add_dynamic_rectangle(self, position, width, height, mass=1):
        """Add a dynamic rectangle that can move and respond to forces, but cannot rotate"""
        inertia = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.friction = 1
        
        # Prevent rotation by setting moment of inertia to infinity
        body.moment = float('inf')
        
        # Create rectangular shape
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.color = pygame.Color('LightSlateGray')
        shape.friction = 0.7
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
