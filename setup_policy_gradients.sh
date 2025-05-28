#!/bin/bash
# Create main directory
mkdir -p reinforcement_learning/policy_gradients
cd reinforcement_learning/policy_gradients

# Create main test files (without docstring)
for i in {0..3}; do
    echo '#!/usr/bin/env python3' > "${i}-main.py"
done

# Create specialized files with docstring
echo -e '#!/usr/bin/env python3\n"""Policy Gradient implementation for reinforcement learning."""' > "policy_gradient.py"
echo -e '#!/usr/bin/env python3\n"""Training module for policy gradient algorithms."""' > "train.py"

# Make all Python files executable
chmod +x *.py

# Create README.md
cat > README.md << 'EOF'
# Policy Gradients

This project implements Policy Gradient methods for reinforcement learning using the Monte-Carlo policy gradient algorithm (REINFORCE).

## Description

Implementation of policy gradient algorithms that directly optimize the policy parameters using gradient ascent on expected rewards. Uses the CartPole-v1 environment from Gymnasium.

## Requirements

- Python 3.9
- Ubuntu 20.04 LTS
- numpy (version 1.25.2)
- gymnasium (version 0.29.1)
- pycodestyle (version 2.11.1)

## Files

- policy_gradient.py - Core policy gradient functions (policy, policy_gradient)
- train.py - Training loop implementation with optional visualization
- 0-main.py - Test simple policy function
- 1-main.py - Test Monte-Carlo policy gradient computation
- 2-main.py - Test training implementation
- 3-main.py - Test training with animation

## Algorithms Implemented

### Policy Function
Computes action probabilities using softmax policy with linear function approximation.

### Monte-Carlo Policy Gradient
Implements REINFORCE algorithm using complete episode returns to estimate policy gradients.

### Training Loop
Full training implementation with:
- Episode-based learning
- Gradient ascent updates
- Score tracking and visualization
- Optional environment rendering

## Usage

Each component can be tested using its corresponding main file:
- ./0-main.py (Test policy function)
- ./1-main.py (Test policy gradient computation)
- ./2-main.py (Test training loop)
- ./3-main.py (Test training with visualization)

## Learning Objectives

- Understanding policy-based reinforcement learning
- Policy gradient theorem and REINFORCE algorithm
- Monte-Carlo methods for policy optimization
- Function approximation in policy space
- Gradient ascent for policy improvement

## Environment

All algorithms are tested on the CartPole-v1 environment, where an agent must balance a pole on a cart by applying left/right forces.

## Mathematical Foundation

The policy gradient theorem states:
∇J(θ) = E[∇log π(a|s,θ) * Q(s,a)]

REINFORCE uses the return G_t as an unbiased estimate of Q(s,a):
∇J(θ) ≈ ∇log π(a_t|s_t,θ) * G_t
EOF

echo "Policy Gradients project structure created successfully!"
echo "Next steps:"
echo "1. Activate conda environment: conda activate deep"
echo "2. Run tests: ./0-main.py, ./1-main.py, etc."
echo "3. Implement remaining functions in policy_gradient.py and train.py" 