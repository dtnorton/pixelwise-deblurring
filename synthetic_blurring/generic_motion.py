# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:17:16 2024

@author: dtnor
"""

import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Disk Moving Around Central Point")

# Colors
black = (0, 0, 0)
red = (255, 0, 0)

# Clock object to control the frame rate
clock = pygame.time.Clock()

# Disk parameters
radius = 20
orbit_radius = 100
center_x, center_y = width // 2, height // 2
angle = 0
speed = 1  # Speed of the disk's movement

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate the new position of the disk
    disk_x = center_x + orbit_radius * math.cos(angle)
    disk_y = center_y + orbit_radius * math.sin(angle)
    angle += speed

    # Clear the screen
    screen.fill(black)

    # Draw the central point
    pygame.draw.circle(screen, red, (center_x, center_y), 5)

    # Draw the moving disk
    pygame.draw.circle(screen, red, (int(disk_x), int(disk_y)), radius)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(1)

pygame.quit()
