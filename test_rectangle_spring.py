#!/usr/bin/env python3

import numpy as np
import pygame
from diffusion_policy.env.spring.spring_env import SpringEnv

def test_rectangle_spring():
    """Test the rectangle spring environment with force display"""
    
    # Create environment
    env = SpringEnv(render_size=512)
    
    # Reset environment
    obs = env.reset()
    
    # Initialize pygame for human rendering
    pygame.init()
    
    print("Testing rectangle spring environment...")
    print("- Dynamic rectangle replaces tee")
    print("- Rectangle cannot rotate (moment = infinity)")
    print("- Force display shows magnitude and direction")
    print("- Success when force reaches 5N")
    print("Press any key in the pygame window or close to exit")
    
    # Run for several steps to see force changes
    for step in range(150):
        # Apply different actions to create various force scenarios
        if step < 40:
            action = np.array([200, 300])  # Move towards rectangle
        elif step < 80:
            action = np.array([300, 300])  # Push rectangle to compress spring
        elif step < 120:
            action = np.array([400, 300])  # Push further to extend spring
        else:
            action = np.array([256, 400])  # Move back to center
            
        obs, reward, done, info = env.step(action)
        
        # Render with human mode to see the force display
        env.render("human")
        
        # Print force and reward information
        if 'total_force_on_tee' in info:
            force = info['total_force_on_tee']
            print(f"Step {step}: Force = {force:.1f}N, Reward = {reward:.3f}")
            
            if force >= 5.0:
                print(f"SUCCESS! Force reached {force:.1f}N (target: 5.0N)")
        
        # Check for pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                print("Key pressed, exiting...")
                env.close()
                return
        
        # Small delay to see the visualization
        pygame.time.wait(80)
        
        if done:
            print(f"Task completed! Final force: {info.get('total_force_on_tee', 0):.1f}N")
            break
    
    print("\nRectangle spring test completed!")
    env.close()

if __name__ == "__main__":
    test_rectangle_spring()
