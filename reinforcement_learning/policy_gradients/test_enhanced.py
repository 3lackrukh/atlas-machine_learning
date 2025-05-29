#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import random
from train_enhanced import train_enhanced

def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

# Create environment
env = gym.make('CartPole-v1')
set_seed(env, 0)

# Run enhanced training - 
print("Starting REINFORCE training with analytics...")
print("Running 10,000 episodes with comprehensive tracking...")
scores, analytics_report = train_enhanced(env, 10000)

# Display results
print(analytics_report)

# Save results
with open('training_analytics.txt', 'w') as f:
    f.write(analytics_report)

print("\nAnalytics saved to 'training_analytics.txt'")
print("Ready to share insights!")
env.close() 