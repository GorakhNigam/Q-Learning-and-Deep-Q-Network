import pygame
import numpy as np
import torch.nn as nn
from types import SimpleNamespace

from DQN import QNetwork
from QLearning import QTable
from utils import MAZE, draw_rewards

def main():
    config = SimpleNamespace()
    config.Learning_method = 'DQN' # Choose from {QLearning, DQN}

    # Setting Maze 
    
    config.MAZE_SIZE = 5 # The size of Maze 
    config.Start_point = [0, 0] # Start point of the agent
    config.End_point = [4, 4] # Destination of the agent
    config.maze_setting = np.zeros((config.MAZE_SIZE, config.MAZE_SIZE), dtype=int)

    # Setting wall in maze
    # Method 1: random walls
    config.wall_number = 4
    np.random.seed(26)
    for _ in range(config.wall_number):
        x = np.random.randint(1, config.MAZE_SIZE)
        y = np.random.randint(1, config.MAZE_SIZE)
        config.maze_setting[x, y] = 1
    
    # Method 2: Freely set walls
    # config.maze_setting[1, 0] = 1 # Set wall at point (1,0)
    
    if(config.maze_setting[config.Start_point[0], config.Start_point[1]] == 1 or 
       config.maze_setting[config.End_point[0], config.End_point[1]] == 1):
        raise ValueError("The start or end point coincides with the obstacle.")
    maze = MAZE(config)

    # Training setting
    # < - TASK V: Hyperparameter Tuning and Evaluation - >
    rewards = []
    running = True
    learning_step = 1
    episodes = 500 # Total number of episodes
    Maximum_steps = 500 # Total number of steps in each episodes
    config.report_frequency = 5 # report average reward every config.report_frequency episode 
    config.pygame_delay = 5 # move delay (ms)
    config.discount_factor = 0.99 # Discount on future rewards
    config.epsilon = 0.9 # Explore Probability

    if(config.Learning_method == 'DQN'):
        config.learning_rate = 1e-4
        config.loss_fn = nn.MSELoss()
        config.capacity = 1000 # Maximum capacity of replay buffer
        config.target_network_update_frequency = 10 # Frequency of updating target network
        config.batch_size = 64
        RLagent = QNetwork(config)

    elif(config.Learning_method == 'QLearning'):
        RLagent = QTable(config)

    else:
        raise ValueError("Incorrect machine learning methods.")
    # < - TASK V: Hyperparameter Tuning and Evaluation - >
    
    print(f"Experiment configs: {config}")

    # Training start
    for episode in range(episodes):
        if not running:
            break
        
        steps = 1
        total_reward = 0
        done_flags = False
        maze.reset()

        while not done_flags and running and steps < Maximum_steps:
            pygame.time.delay(config.pygame_delay)
            steps += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break
            
            pos, action, new_pos, reward, done_flags = RLagent.step(maze.robot_pos)
            total_reward += reward

            if(config.Learning_method == 'DQN'):
                RLagent.replay_buffer.push(pos, action, new_pos, reward, done_flags)
                if len(RLagent.replay_buffer) > config.batch_size:
                    learning_step += 1
                    RLagent.learn()
                if learning_step % config.target_network_update_frequency == 0:
                    RLagent.target_update()

            elif(config.Learning_method == 'QLearning'):
                RLagent.update_q_table(pos, action, new_pos, reward, done_flags)

            maze.move_Robot(new_pos)

        RLagent.epsilon_decline()
        rewards.append(total_reward)
        if episode % config.report_frequency == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(rewards[-config.report_frequency:])}, Epsilon: {RLagent.epsilon:.2f}")

    maze.close_pygame()
    draw_rewards(rewards)

if __name__ == "__main__":
    main()