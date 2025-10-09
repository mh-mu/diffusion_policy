#!/usr/bin/env python3

"""
Debug script to check spring forces in SpringEnv
"""

import sys
import os
sys.path.insert(0, '/home/mei/workspace/diffusion_policy')

import numpy as np
import pymunk
from diffusion_policy.env.spring.spring_env import SpringEnv

def debug_spring_forces():
    print("=== Debugging Spring Forces ===")
    
    # Create environment
    env = SpringEnv(render_size=96)
    
    # Reset to initialize physics
    obs = env.reset()
    print(f"Environment initialized")
    print(f"Agent body type: {env.agent.body_type}")  # Should be KINEMATIC (1)
    print(f"Block body type: {env.block.body_type}")  # Should be DYNAMIC (2) 
    print(f"Spring anchor body type: {env.spring_anchor.body_type}")  # Should be DYNAMIC (2)
    
    print(f"\nInitial positions:")
    print(f"Agent: {env.agent.position}")
    print(f"Block (tee): {env.block.position}")  
    print(f"Spring anchor: {env.spring_anchor.position}")
    
    # Check spring constraint properties
    print(f"\nSpring constraint:")
    print(f"Rest length: {env.spring_constraint.rest_length}")
    print(f"Stiffness: {env.spring_constraint.stiffness}")
    print(f"Damping: {env.spring_constraint.damping}")
    
    # Check initial spring state
    spring_vector = env.spring_anchor.position - env.block.position
    current_length = spring_vector.length
    print(f"Initial spring length: {current_length:.2f}")
    print(f"Initial displacement: {current_length - env.spring_constraint.rest_length:.2f}")
    
    print(f"\n=== Testing Spring Dynamics ===")
    
    # Test 1: Push agent towards tee to compress spring
    print("\nTest 1: Push agent towards tee...")
    for i in range(5):
        # Move agent towards the tee  
        action = np.array([256.0, 320.0])  # Move agent closer to tee
        obs, reward, done, info = env.step(action)
        
        if 'spring_length' in info:
            print(f"Step {i+1}:")
            print(f"  Agent pos: {info['pos_agent']}")
            print(f"  Tee pos: {info['block_pose'][:2]}")
            print(f"  Spring anchor pos: {info['spring_anchor_pos']}")
            print(f"  Spring length: {info['spring_length']:.2f}")
            print(f"  Spring displacement: {info['spring_displacement']:.2f}")
            print(f"  Spring force: {info['spring_force']:.2f}")
            print(f"  Spring compressed: {info['spring_compressed']}")
        
        # Check velocities to see if objects are moving
        print(f"  Agent velocity: {np.array(env.agent.velocity)}")
        print(f"  Tee velocity: {np.array(env.block.velocity)}")
        print(f"  Spring anchor velocity: {np.array(env.spring_anchor.velocity)}")
        print()
    
    # Test 2: Let physics run without agent control
    print("Test 2: Let physics run without agent input...")
    for i in range(5):
        obs, reward, done, info = env.step(None)  # No action
        
        if 'spring_length' in info:
            print(f"Free step {i+1}:")
            print(f"  Spring length: {info['spring_length']:.2f}")
            print(f"  Spring force: {info['spring_force']:.2f}")
            print(f"  Tee velocity: {np.array(env.block.velocity)}")
            print(f"  Spring anchor velocity: {np.array(env.spring_anchor.velocity)}")
    
    env.close()
    print("Debug complete!")

if __name__ == "__main__":
    debug_spring_forces()
