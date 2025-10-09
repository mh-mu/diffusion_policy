#!/usr/bin/env python3

"""
Test script for the SpringEnv to verify it works correctly
"""

import sys
import os
sys.path.insert(0, '/home/mei/workspace/diffusion_policy')

import numpy as np
from diffusion_policy.env.spring.spring_env import SpringEnv

def test_spring_env():
    print("Testing SpringEnv...")
    
    # Create environment
    env = SpringEnv(target_force=5.0, render_size=96)
    
    # Test reset
    print("Testing reset...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test step
    print("\nTesting steps...")
    for i in range(10):
        # Random action (move towards spring)
        action = np.array([350.0, 250.0])  # Move towards the spring
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        print(f"  Force: {info['current_force']:.2f}N / {info['target_force']:.2f}N")
        print(f"  Spring length: {info['spring_length']:.1f} (natural: {info['natural_length']:.1f})")
        
        if done:
            print("  SUCCESS: Target force achieved!")
            break
    
    # Test rendering
    print("\nTesting rendering...")
    img = env.render('rgb_array')
    print(f"Rendered image shape: {img.shape}")
    
    env.close()
    print("SpringEnv test completed successfully!")

if __name__ == "__main__":
    test_spring_env()
