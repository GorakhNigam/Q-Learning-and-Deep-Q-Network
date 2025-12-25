import pygame
import numpy as np
import matplotlib.pyplot as plt

class MAZE():
    def __init__(self, config):
        self.MAZE_SIZE = config.MAZE_SIZE
        self.maze = config.maze_setting

        self.Start_point = config.Start_point
        self.End_point = config.End_point

        self.GRID_SIZE = 50
        self.Block_Color = (0, 0, 0)
        self.Line_Color = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.Target_Color = (255, 0, 0)
        self.Destination_color = (255, 255, 0)
        self.WINDOW_WIDTH = self.MAZE_SIZE * self.GRID_SIZE
        self.WINDOW_HEIGHT = self.MAZE_SIZE * self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        self.reset()

    def initialize_maze(self):        
        pygame.init()
        self.screen.fill(self.WHITE)
        
        for i in range(self.MAZE_SIZE + 1):
            pygame.draw.line(self.screen, self.Line_Color, (i * self.GRID_SIZE, 0), (i * self.GRID_SIZE, self.MAZE_SIZE * self.GRID_SIZE))
            pygame.draw.line(self.screen, self.Line_Color, (0, i * self.GRID_SIZE), (self.MAZE_SIZE * self.GRID_SIZE, i * self.GRID_SIZE))
        
        for x in range(self.MAZE_SIZE):
            for y in range(self.MAZE_SIZE):
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, self.Block_Color, 
                                (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        
        start_rect = pygame.Rect(self.Start_point[0]*self.GRID_SIZE, self.Start_point[1]*self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.Target_Color, start_rect, width=2)
        
        end_center = (self.End_point[0]*self.GRID_SIZE + self.GRID_SIZE//2, 
                    self.End_point[1]*self.GRID_SIZE + self.GRID_SIZE//2)
        end_radius = self.GRID_SIZE // 2 - 1
        pygame.draw.circle(self.screen, self.Destination_color, end_center, end_radius)
        pygame.display.set_caption("Maze")

    def move_Robot(self, robot_pos):
        self.robot_pos = robot_pos
        self.initialize_maze()
        if robot_pos:
            robot_rect = (robot_pos[0]*self.GRID_SIZE, 
                        robot_pos[1]*self.GRID_SIZE, 
                        self.GRID_SIZE, 
                        self.GRID_SIZE)

        pygame.draw.rect(self.screen, self.Target_Color, robot_rect)
        pygame.display.update()
    
    def close_pygame(self):
        pygame.quit()

    def reset(self):
        self.initialize_maze()
        self.move_Robot(self.Start_point)

def draw_rewards(rewards, window = 10):
    rewards = np.convolve(rewards, np.ones(window), "valid") / window
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('Result.jpg')
    plt.show()
    print("Results have been saved at 'Result.jpg'.")
