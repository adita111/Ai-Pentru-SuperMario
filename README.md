# Super Mario Bros AI Agent — Reinforcement Learning with PPO

## Overview

This project was developed as my Bachelor's thesis and explores the application of Reinforcement Learning to train an autonomous AI agent to play Super Mario Bros.

The agent learns through interaction with the game environment, using Proximal Policy Optimization (PPO) and a Convolutional Neural Network (CNN) with an Actor-Critic architecture.

The project includes a complete pipeline for environment preprocessing, agent training, reward shaping, performance evaluation, and visualization of training results.

## Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch
- **Reinforcement Learning:** Proximal Policy Optimization (PPO)
- **Neural Network:** CNN with Actor-Critic architecture
- **Environment:** Gymnasium, gym-super-mario-bros, NES-Py
- **Data Analysis:** Training metrics and performance visualizations

## How It Works

### 1. Environment Preprocessing

Game frames are processed before being passed to the neural network:

- Convert RGB frames to grayscale.
- Resize images to 84×84 pixels.
- Stack 4 consecutive frames to capture movement.
- Apply frame skipping to improve training efficiency.

### 2. Neural Network Architecture

The agent uses a CNN with an Actor-Critic architecture.

- **CNN:** Extracts relevant visual features from game frames.
- **Actor:** Produces a probability distribution over available actions.
- **Critic:** Estimates the value of the current game state.

### 3. Training with PPO

The agent learns through repeated interactions with the environment using Proximal Policy Optimization.

Custom reward shaping encourages forward movement, exploration, and survival while penalizing unnecessary backward movement, stagnation, and deaths.

## Training and Evaluation

Training performance is monitored using metrics such as:

- Total training steps
- Distance reached in the level
- Episode rewards
- Agent performance during evaluation

The project also includes a comparison with a Double Deep Q-Network (DDQN) approach.

## Challenges

Some of the main challenges included:

- Designing a reward function that encourages meaningful progress.
- Reducing repetitive behavior and unnecessary deaths.
- Balancing exploration and exploitation.
- Training a computationally demanding reinforcement learning model without GPU acceleration.

## Future Improvements

- Train the agent using GPU acceleration.
- Experiment with alternative reward functions and hyperparameters.
- Improve generalization across multiple levels.
- Compare additional reinforcement learning algorithms.
