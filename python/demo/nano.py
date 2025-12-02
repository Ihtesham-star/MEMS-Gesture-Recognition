import socket
import struct
import time
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

class NanoParticle:
    def __init__(self, position, size=0.05, color=(1.0, 0.0, 0.0)):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.color = color
        self.is_grabbed = False
        self.original_position = np.array(position, dtype=np.float32)

class TargetRegion:
    def __init__(self, position, size=0.2, color=(0.0, 1.0, 0.0, 0.3)):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.color = color
        self.particles_inside = 0

class LaserPointer:
    def __init__(self, position=(0, 0, 0), color=(1.0, 1.0, 0.0)):
        self.position = np.array(position, dtype=np.float32)
        self.color = color
        self.speed = 0.05
        self.grabbed_particle = None
        
    def move(self, direction):
        self.position += direction * self.speed
        # Clamp positions to keep within the visible area
        self.position = np.clip(self.position, -1.0, 1.0)
        
        # Update grabbed particle position if any
        if self.grabbed_particle:
            self.grabbed_particle.position = self.position.copy()
    
    def grab_particle(self, particles, distance_threshold=0.1):
        if self.grabbed_particle:
            return False
            
        for particle in particles:
            if not particle.is_grabbed:
                dist = np.linalg.norm(self.position - particle.position)
                if dist < distance_threshold:
                    particle.is_grabbed = True
                    self.grabbed_particle = particle
                    return True
        return False
    
    def release_particle(self):
        if self.grabbed_particle:
            self.grabbed_particle.is_grabbed = False
            self.grabbed_particle = None
            return True
        return False

class Simulation:
    def __init__(self):
        # Initialize Pygame and OpenGL
        pygame.init()
        self.display_width = 800
        self.display_height = 600
        pygame.display.set_mode((self.display_width, self.display_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Nanoparticle Manipulation Simulation")
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        gluPerspective(45, (self.display_width / self.display_height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -3)
        
        # Setup TCP client for laser control
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.connect(('localhost', 30001))
            self.laser_connected = True
            print("Connected to laser control server")
        except Exception as e:
            self.laser_connected = False
            print(f"Failed to connect to laser control server: {e}")
            print("Running in simulation-only mode")
        
        # Initialize simulation objects
        self.laser = LaserPointer()
        self.particles = [
            NanoParticle((random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), random.uniform(-0.5, 0.5)))
            for _ in range(5)
        ]
        self.target = TargetRegion((0.7, 0.7, 0))
        
        # MEMS mirror normalization factors
        self.screen_to_mems_x = 1.0
        self.screen_to_mems_y = 1.0
        
        # Game state
        self.running = True
        self.score = 0
        self.timer = 60  # 60 seconds game time
        self.last_time = time.time()
        self.game_over = False
        
        # Font for text display
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_g:
                    self.handle_grab()
                elif event.key == pygame.K_r:
                    self.handle_release()
                elif event.key == pygame.K_SPACE and self.game_over:
                    self.reset_game()
        
        # Continuous movement with key presses
        keys = pygame.key.get_pressed()
        movement = np.zeros(3, dtype=np.float32)
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement[0] -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement[0] += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement[1] += 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement[1] -= 1
        if keys[pygame.K_q]:  # Move deeper
            movement[2] -= 1
        if keys[pygame.K_e]:  # Move closer
            movement[2] += 1
            
        # Normalize movement vector if not zero
        if np.any(movement):
            movement = movement / np.linalg.norm(movement) * self.laser.speed
            self.laser.move(movement)
            
            # Send position to MEMS mirror if connected
            if self.laser_connected:
                # Convert from OpenGL coordinates to MEMS mirror coordinates
                mems_x = self.laser.position[0] * self.screen_to_mems_x
                mems_y = self.laser.position[1] * self.screen_to_mems_y
                
                try:
                    self.s.sendall(struct.pack('dd', mems_x, mems_y))
                except Exception as e:
                    print(f"Failed to send position to MEMS mirror: {e}")
                    self.laser_connected = False
    
    def handle_grab(self):
        if self.laser.grab_particle(self.particles):
            print("Particle grabbed")
            if self.laser_connected:
                try:
                    self.s.sendall(b'pinch')
                except Exception as e:
                    print(f"Failed to send pinch command: {e}")
    
    def handle_release(self):
        if self.laser.release_particle():
            print("Particle released")
            if self.laser_connected:
                try:
                    self.s.sendall(b'release')
                except Exception as e:
                    print(f"Failed to send release command: {e}")
            
            # Check if particle is in target region
            for particle in self.particles:
                if not particle.is_grabbed:
                    dist = np.linalg.norm(particle.position - self.target.position)
                    if dist < self.target.size:
                        self.score += 1
                        self.target.particles_inside += 1
                        
                        # Reset particle to a new random position
                        particle.position = np.array([
                            random.uniform(-0.8, 0.8),
                            random.uniform(-0.8, 0.8),
                            random.uniform(-0.5, 0.5)
                        ], dtype=np.float32)
                        particle.original_position = particle.position.copy()
    
    def update(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if not self.game_over:
            self.timer -= dt
            if self.timer <= 0:
                self.game_over = True
    
    def draw_text(self, text, position, color=(255, 255, 255)):
        text_surface = self.font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_width, text_height = text_surface.get_size()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glWindowPos2d(position[0], position[1])
        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw tissue/body environment (simplified as a cube)
        glPushMatrix()
        glColor4f(0.8, 0.8, 0.9, 0.3)  # Semi-transparent light blue/gray
        self.draw_cube(1.0)
        glPopMatrix()
        
        # Draw target region
        glPushMatrix()
        glTranslatef(self.target.position[0], self.target.position[1], self.target.position[2])
        glColor4f(*self.target.color)
        self.draw_cube(self.target.size)
        glPopMatrix()
        
        # Draw particles
        for particle in self.particles:
            glPushMatrix()
            glTranslatef(particle.position[0], particle.position[1], particle.position[2])
            glColor3f(*particle.color)
            self.draw_sphere(particle.size, 16, 16)
            glPopMatrix()
        
        # Draw laser pointer
        glPushMatrix()
        glTranslatef(self.laser.position[0], self.laser.position[1], self.laser.position[2])
        glColor3f(*self.laser.color)
        self.draw_sphere(0.03, 8, 8)
        glPopMatrix()
        
        # Draw laser beam (line from origin to laser position)
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 0.0)  # Yellow beam
        glVertex3f(0, 0, 0)
        glVertex3f(self.laser.position[0], self.laser.position[1], self.laser.position[2])
        glEnd()
        
        # Draw UI elements
        self.draw_text(f"Score: {self.score}", (10, self.display_height - 30))
        self.draw_text(f"Time: {int(self.timer)}s", (10, self.display_height - 60))
        
        if self.game_over:
            self.draw_text("GAME OVER - Press SPACE to restart", 
                          (self.display_width//2 - 180, self.display_height//2), 
                          (255, 0, 0))
        
        # Draw controls help
        self.draw_text("Controls: Arrow keys to move, G to grab, R to release", 
                      (10, 10))
        
        pygame.display.flip()
    
    def draw_sphere(self, radius, slices, stacks):
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
    
    def draw_cube(self, size):
        half_size = size / 2
        
        # Draw a cube with given size
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        # Back face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        # Top face
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        # Bottom face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        # Right face
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        # Left face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glEnd()
    
    def reset_game(self):
        # Reset game state
        self.score = 0
        self.timer = 60
        self.game_over = False
        self.target.particles_inside = 0
        
        # Reset particles
        for particle in self.particles:
            particle.position = np.array([
                random.uniform(-0.8, 0.8),
                random.uniform(-0.8, 0.8),
                random.uniform(-0.5, 0.5)
            ], dtype=np.float32)
            particle.original_position = particle.position.copy()
            particle.is_grabbed = False
        
        # Reset laser
        self.laser.position = np.array([0, 0, 0], dtype=np.float32)
        self.laser.grabbed_particle = None
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw_scene()
            pygame.time.delay(30)  # ~30 FPS
        
        # Clean up
        if self.laser_connected:
            try:
                self.s.sendall(b'release')  # Make sure to release any grab state
                self.s.close()
            except:
                pass
        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()