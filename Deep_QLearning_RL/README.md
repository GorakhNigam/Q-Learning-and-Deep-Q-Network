## Overview
This tutorial provides a guide to configuring your Python environment for Q-Learning and Deep Q-Learning. 
We recommend [Anaconda](https://www.anaconda.com/) for its robust package management capabilities.

For installing please refer to [Anaconda download page](https://www.anaconda.com/download), and referring [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda) to started conda.

## Step 1, install Python requirements:

### Create New "RL" Environment:
For Windows: use Anaconda Prompt; for macOS and Linux: use Terminal.
- ``cd xxx/Maze``
- ``conda create -y -n RL python=3.9``
### Activate the "RL" Environment:
- ``conda activate RL``
### Install Dependencies:
- ``pip install -r requirement.txt``

## Step 2, run the experiment:
- ``python main.py``