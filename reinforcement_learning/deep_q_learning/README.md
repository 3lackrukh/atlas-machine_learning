🎮 Deep Q-Learning for Atari's Breakout

This project implements a Deep Q-Network (DQN) agent that can play Atari's Breakout. The implementation features scaled reward shaping, frame stacking, and episodic target network updates to accelerate learning.
Environment Setup
Prerequisites

    Ubuntu 20.04 LTS
    Anaconda or Miniconda (Python package manager)
    VSCode (optional, for development)

⚙️ Setting up the Conda Environment

    Install Anaconda (if not already installed):
```
    bash

    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
    bash Anaconda3-2023.09-0-Linux-x86_64.sh
    source ~/.bashrc
```

    Create the environment from the provided YAML file:
```
    bash

    conda env create -f environment.yml
```

    Activate the environment:
```
    bash

    conda activate atari-dqn
```

Key Dependencies

    Python 3.9: Base programming language 🐍
    gymnasium[atari]==0.29.1: OpenAI's Gymnasium library with Atari environments
    tensorflow==2.15.0 & keras==2.15.0: Deep learning frameworks
    keras-rl2==1.0.4: Reinforcement learning library for Keras
    autorom[accept-rom-license]: Package for Atari ROM management

NOTE:

   After setting and entering the conda environment
   please take care to run the fix_keras-rl.py script to
   patch an attribute bug in Keras 2.15.0

🔧 Troubleshooting

    If encountering ROM licensing issues, run: AutoROM --accept-license
    For visualization problems, ensure you have the appropriate display drivers
    If tensorflow has compatibility issues, verify CUDA and cuDNN versions (if using GPU)

Project Structure


├── environment.yml       # Conda environment config file
├── fix_keras_rl.py       # A patch to help Keras play well with Gymnasium
├── train.py              # Script to train the Deep Q-learning agent
├── play.py               # Script to visualize the trained agent playing Breakout
├── policy.h5             # Saved model weights (created by train.py)
└── utils/                # Utility modules for DQN implementation
    ├── __init__.py       
    ├── callbacks.py      # Custom callbacks for training (e.g., episodic target updates)
    ├── models.py         # Neural network architecture definition
    ├── patching.py       # Utilities for patching DQNAgent for continuous training
    ├── processors.py     # Observation and reward processors
    └── wrappers.py       # Environment wrappers for Gymnasium compatibility

🚀 Usage

    Train the agent:
```
    bash

    python train.py
```
    This will train the DQN agent and save the model weights as policy.h5.
    Training may take several hours depending on your hardware.

    Watch the trained agent play:
```
# Default usage (uses policy.h5 in the current directory)
python play.py

# Using weights from a specific experiment
python play.py --weights notebook/V1/policy_100000.h5

# Specify both weights file and number of episodes
python play.py --weights notebook/V2/policy_200000.h5 --episodes 10
```
    This loads the trained policy and visualizes the agent playing Breakout.

🧠 Implementation Details

This project implements a Deep Q-Network (DQN) agent with several training enhancement features:
Core DQN Components

    Double DQN: Reduces value overestimation by using separate networks for action selection and evaluation
    Experience Replay: Stores and reuses past experiences to break correlations between consecutive samples
    Frame Stacking: Combines multiple frames to capture motion information
    Target Network: Stabilizes training by updating the target network episodically rather than continuously

Advanced Features

    Adaptive Reward Shaping: Dynamically scales rewards based on the best performance seen so far
    Terminal State Handling: Provides appropriate negative feedback at episode termination
    Survival Bonus: Encourages the agent to stay alive longer, creating a more varied reward landscape
    Episodic Target Updates: Updates the target network after a fixed number of episodes rather than steps
    Continuous Training Support: Allows training to resume without redoing warmup steps

📊 Colab Notebook and Experiments

The complete implementation is available in a Colab notebook. 
The notebook includes:

    Training Monitoring: Real-time visualization of rewards, Q-values, and episode lengths
    Analytics Dashboard: Tools for analyzing training performance


    Multiple Versions: Different iterations (V1-V5) with progressive improvements:
        V1: Baseline DQN implementation
        V2: Implements adaptive reward shaping
        V3: Target network updates every 100 episodes
        V4: Reduces no-operation initial steps from 30 to 7
        V5: Episodic weight update reduced from 100 to 7
            - Scaled rewards, negative rewards clipped to -1.0

Each version folder contains:

   - A PDF copy of the notebook configuration it was trained in

   - Trained network weights 
      -- naming convention for weights files is:
         policy_100000.h5
            Where the number specifies the
            number of steps that model was trained for
      -- Each experiment was trained at least 100000 steps
         to get a sense of the strength of the training dynamics

   - logs/
      -- A folder containing training stats and progress visualizations
         for each 100000 step training session
      

       

📚 References

    Deep Q-Learning - Combining Neural Networks and Reinforcement Learning
    Human-level control through deep reinforcement learning - Original DQN paper
    keras-rl2 Documentation
    Gymnasium Documentation
