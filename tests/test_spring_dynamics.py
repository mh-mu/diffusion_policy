#!/usr/bin/env python3

"""
Test script for the SpringEnv with dynamic circle and spring constraint
"""

import sys
import os
sys.path.insert(0, '/home/mei/workspace/diffusion_policy')

import numpy as np
from diffusion_policy.env.spring.spring_env import SpringEnv

def test_spring_with_dynamic_circle():
    print("Testing SpringEnv with dynamic circle and spring...")
    
    # Create environment
    env = SpringEnv(render_size=96)
    
    # Test reset
    print("Testing reset...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Check if spring anchor was created
    if hasattr(env, 'spring_anchor') and env.spring_anchor is not None:
        print(f"Spring anchor position: {env.spring_anchor.position}")
        print(f"Spring anchor is dynamic: {env.spring_anchor.body_type == 2}")  # 2 = DYNAMIC
    else:
        print("ERROR: Spring anchor not created!")
        return
    
    # Check if spring constraint exists
    if hasattr(env, 'spring_constraint') and env.spring_constraint is not None:
        print(f"Spring constraint rest length: {env.spring_constraint.rest_length}")
        print(f"Spring constraint stiffness: {env.spring_constraint.stiffness}")
        print(f"Spring constraint damping: {env.spring_constraint.damping}")
    else:
        print("ERROR: Spring constraint not created!")
        return
        
    # Test a few steps to see spring behavior
    print("\nTesting spring dynamics...")
    for i in range(10):
        # Push the tee towards the spring anchor
        action = np.array([300.0, 300.0])  # Move towards the spring
        obs, reward, done, info = env.step(action)
        
        if 'spring_length' in info:
            print(f"Step {i+1}:")
            print(f"  Tee position: {info['block_pose'][:2]}")
            print(f"  Spring anchor position: {info['spring_anchor_pos']}")
            print(f"  Spring length: {info['spring_length']:.2f} (rest: {info['spring_rest_length']:.2f})")
            print(f"  Spring force: {info['spring_force']:.2f}")
        else:
            print(f"Step {i+1}: No spring info available")
    
    # Test rendering
    print("\nTesting rendering with spring visualization...")
    try:
        img = env.render('rgb_array')
        print(f"Rendered image shape: {img.shape}")
        print("Rendering successful!")
    except Exception as e:
        print(f"Rendering failed: {e}")
    
    env.close()
    print("Spring dynamics test completed!")

if __name__ == "__main__":
    test_spring_with_dynamic_circle()
