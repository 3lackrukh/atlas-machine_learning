# Temporal Difference Learning

This project implements various temporal difference learning algorithms for reinforcement learning, focusing on model-free prediction and control methods.

## Description

Implementation of key temporal difference algorithms including Monte Carlo methods, TD(lambda), and SARSA(lambda) using the FrozenLake8x8-v1 environment from Gymnasium.

## Requirements

- Python 3.9
- Ubuntu 20.04 LTS
- numpy (version 1.25.2)
- gymnasium (version 0.29.1)
- pycodestyle (version 2.11.1)

## Files

- 0-monte_carlo.py - Monte Carlo algorithm implementation
- 1-td_lambtha.py - TD(lambda) algorithm with eligibility traces  
- 2-sarsa_lambtha.py - SARSA(lambda) on-policy control algorithm

## Algorithms Implemented

### Monte Carlo
Performs Monte Carlo value estimation using complete episode returns.

### TD(lambda)
Temporal difference learning with eligibility traces, combining benefits of TD and Monte Carlo methods.

### SARSA(lambda)
State-Action-Reward-State-Action learning with eligibility traces for on-policy control.

## Usage

Each algorithm can be tested using its corresponding main file:
- ./0-main.py (Test Monte Carlo)
- ./1-main.py (Test TD lambda)  
- ./2-main.py (Test SARSA lambda)

## Learning Objectives

- Monte Carlo methods for value estimation
- Temporal difference learning and bootstrapping
- Eligibility traces and n-step methods
- On-policy vs off-policy learning
- SARSA algorithm variants

## Environment

All algorithms are tested on the FrozenLake8x8-v1 environment, a grid world where an agent must navigate from start to goal while avoiding holes.
