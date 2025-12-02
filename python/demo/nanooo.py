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
    def __init__(self, position, size=0.02, color=None):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        
        # Assign a realistic color if none provided (gold, silver, or copper-like for nanoparticles)
        if color is None:
            colors = [
                (0.85, 0.65, 0.13),  # Gold
                (0.75, 0.75, 0.75),  # Silver
                (0.72, 0.45, 0.20),  # Copper
                (0.55, 0.55, 0.87),  # Metallic blue
                (0.60, 0.80, 0.60),  # Metallic green
            ]
            self.color = random.choice(colors)
        else:
            self.color = color
            
        self.is_grabbed = False
        self.original_position = np.array(position, dtype=np.float32)
        # Add velocity for more realistic movement
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Add small random movement
        self.brownian_factor = 0.0005

class TargetRegion:
    def __init__(self, position, size=0.12, color=(0.0, 0.6, 0.2, 0.4)):
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.color = color
        self.particles_inside = 0
        # Pulsating effect
        self.pulse_factor = 0
        self.pulse_speed = 0.05

class LaserPointer:
    def __init__(self, position=(0, 0, 0), color=(1.0, 0.2, 0.2)):
        self.position = np.array(position, dtype=np.float32)
        self.color = color
        self.speed = 0.03
        self.grabbed_particle = None
        self.size = 0.04  # Laser pointer size
        self.beam_width = 0.01  # Width of the laser beam
        self.beam_intensity = 0.8  # Brightness of beam
        
    def move(self, direction):
        self.position += direction * self.speed
        # Clamp positions to keep within the visible area
        self.position = np.clip(self.position, -0.9, 0.9)
        
        # Update grabbed particle position if any
        if self.grabbed_particle:
            self.grabbed_particle.position = self.position.copy()
    
    def grab_particle(self, particles, distance_threshold=0.08):
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

class BiologicalTissue:
    def __init__(self):
        # Create a more complex tissue structure
        self.cells = []
        self.vessels = []
        self.ecm_particles = []  # Extracellular matrix particles
        
        # Generate some cell positions
        for _ in range(15):
            pos = np.array([
                random.uniform(-0.9, 0.9),
                random.uniform(-0.9, 0.9),
                random.uniform(-0.9, 0.9)
            ])
            size = random.uniform(0.05, 0.15)
            transparency = random.uniform(0.1, 0.3)
            # Cell colors range from light pink to light tan
            color = (
                random.uniform(0.8, 0.95),
                random.uniform(0.7, 0.85),
                random.uniform(0.7, 0.85),
                transparency
            )
            self.cells.append((pos, size, color))
            
        # Generate some vessel paths
        for _ in range(5):
            start = np.array([
                random.uniform(-0.9, 0.9),
                random.uniform(-0.9, 0.9),
                random.uniform(-0.9, 0.9)
            ])
            end = np.array([
                random.uniform(-0.9, 0.9),
                random.uniform(-0.9, 0.9),
                random.uniform(-0.9, 0.9)
            ])
            radius = random.uniform(0.02, 0.04)
            # Vessel colors from red to blue for arteries/veins
            if random.random() > 0.5:
                color = (0.7, 0.1, 0.1, 0.5)  # Red for arteries
            else:
                color = (0.1, 0.1, 0.7, 0.5)  # Blue for veins
            self.vessels.append((start, end, radius, color))
            
        # Generate ECM particles
        for _ in range(100):
            pos = np.array([
                random.uniform(-0.95, 0.95),
                random.uniform(-0.95, 0.95),
                random.uniform(-0.95, 0.95)
            ])
            size = random.uniform(0.005, 0.015)
            # ECM colors from white to light yellow
            color = (
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 0.9),
                random.uniform(0.6, 0.8),
                random.uniform(0.2, 0.5)
            )
            self.ecm_particles.append((pos, size, color))

class Simulation:
    def __init__(self):
        # Initialize Pygame and OpenGL
        pygame.init()
        self.display_width = 1920
        self.display_height = 1080
        pygame.display.set_mode((self.display_width, self.display_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Nanoparticle Manipulation Simulation - Educational Tool")
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        light_ambient = [0.8, 0.8, 0.8, 1.0]
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_position = [1.0, 1.0, 2.0, 0.0]
        
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
   
        # Switch to the projection matrix to set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.display_width / self.display_height, 0.1, 50.0)

        # Switch back to the modelview matrix for view transformations 
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3)
        glRotatef(20, 1, 0, 0)

        
        # gluPerspective(45, (self.display_width / self.display_height), 0.1, 50.0)
        # glTranslatef(0.0, 0.0, -3)
        
        # Initial camera rotation for better view
        # glRotatef(20, 1, 0, 0)
        
        # Setup TCP client for laser control
        self.laser_connected = False
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect(('localhost', 30001))
            self.laser_connected = True
            print("Connected to laser control server")
        except Exception as e:
            print(f"Failed to connect to laser control server: {e}")
            print("Running in simulation-only mode")
        
        # Initialize simulation objects
        self.laser = LaserPointer(color=(1.0, 0.2, 0.2))  # Red laser
        
        # Create nanoparticles at random positions
        self.particles = [
            NanoParticle(
                (random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), random.uniform(-0.5, 0.5)),
                size=0.02  # Smaller particles
            )
            for _ in range(12)
        ]
        
        # Create target regions (could be cellular targets, tumors, etc.)
        self.targets = [
            TargetRegion((0.7, 0.7, 0), size=0.12, color=(0.0, 0.6, 0.2, 0.4)),  # Green target
            TargetRegion((-0.7, 0.5, 0.3), size=0.15, color=(0.6, 0.2, 0.6, 0.4))  # Purple target
        ]
        
        # Create biological tissue environment
        self.tissue = BiologicalTissue()
        
        # MEMS mirror normalization factors
        self.screen_to_mems_x = 1.0
        self.screen_to_mems_y = 1.0
        
        # Simulation state
        self.running = True
        self.particles_delivered = 0
        self.last_time = time.time()
        self.rotation_x = 0
        self.rotation_y = 0
        self.show_help = True
        self.paused = False
        
        # Camera settings
        self.camera_distance = 3.0
        self.camera_rotation_x = 20
        self.camera_rotation_y = 0
        
        # Info display settings
        self.info_mode = 0  # 0: basic, 1: detailed, 2: educational
        self.educational_text = [
            "Nanoparticles can be manipulated using optical forces",
            "This is known as optical tweezers or optical trapping",
            "It allows precise positioning of particles at microscale",
            "Applications include targeted drug delivery",
            "And precision manipulation of biological molecules",
            "Particles experience both scattering and gradient forces",
            "Gradient forces pull particles toward beam center",
            "This technique was pioneered by Arthur Ashkin"
        ]
        self.info_timer = 0
        self.current_info_text = 0
        
        # Font for text display
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 18)
        self.large_font = pygame.font.SysFont('Arial', 24)
        
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
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_i:
                    # Cycle through info display modes
                    self.info_mode = (self.info_mode + 1) % 3
                # Camera control with WASD + QE
                elif event.key == pygame.K_1:
                    # Reset view
                    self.camera_rotation_x = 20
                    self.camera_rotation_y = 0
                    self.camera_distance = 3.0
                    
        # Get mouse position for camera rotation when right button is pressed
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[2]:  # Right mouse button
            mouse_rel = pygame.mouse.get_rel()
            self.camera_rotation_y += mouse_rel[0] * 0.5
            self.camera_rotation_x += mouse_rel[1] * 0.5
            # Clamp vertical rotation to avoid flipping
            self.camera_rotation_x = np.clip(self.camera_rotation_x, -90, 90)
        else:
            pygame.mouse.get_rel()  # Reset relative movement
            
        # Mouse wheel for zoom
        for event in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            if event.button == 4:  # Scroll up
                self.camera_distance = max(1.5, self.camera_distance - 0.1)
            elif event.button == 5:  # Scroll down
                self.camera_distance = min(10.0, self.camera_distance + 0.1)
        
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
        if keys[pygame.K_z]:  # Move deeper
            movement[2] -= 1
        if keys[pygame.K_x]:  # Move closer
            movement[2] += 1
            
        # Normalize movement vector if not zero
        if np.any(movement) and not self.paused:
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
        if not self.paused and self.laser.grab_particle(self.particles):
            print("Particle grabbed")
            if self.laser_connected:
                try:
                    self.s.sendall(b'pinch')
                except Exception as e:
                    print(f"Failed to send pinch command: {e}")
    
    def handle_release(self):
        if not self.paused and self.laser.release_particle():
            print("Particle released")
            if self.laser_connected:
                try:
                    self.s.sendall(b'release')
                except Exception as e:
                    print(f"Failed to send release command: {e}")
            
            # Check if particle is in target region
            for particle in self.particles:
                if not particle.is_grabbed:
                    for target in self.targets:
                        dist = np.linalg.norm(particle.position - target.position)
                        if dist < target.size:
                            self.particles_delivered += 1
                            target.particles_inside += 1
                            
                            # Reset particle to a new random position
                            # particle.position = np.array([
                               # random.uniform(-0.8, 0.8),
                               # random.uniform(-0.8, 0.8),
                               # random.uniform(-0.5, 0.5)
                            # ], dtype=np.float32)
                            # particle.original_position = particle.position.copy()
                            break
    
    def update(self, dt):
        if self.paused:
            return
            
        # Update educational info text timer
        self.info_timer += dt
        if self.info_timer > 5.0:  # Change text every 5 seconds
            self.info_timer = 0
            self.current_info_text = (self.current_info_text + 1) % len(self.educational_text)
            
        # Update target region pulsing effect
        for target in self.targets:
            target.pulse_factor += target.pulse_speed * dt
            if target.pulse_factor > 1.0:
                target.pulse_factor = 0.0
        
        # Update particles - add some Brownian motion to free particles
        for particle in self.particles:
            if not particle.is_grabbed:
                # Apply small random movement (Brownian motion)
                # brownian_force = np.array([
                    # random.uniform(-1, 1),
                    # random.uniform(-1, 1),
                    # random.uniform(-1, 1)
                # ]) * particle.brownian_factor
                
                # particle.velocity = particle.velocity * 0.98 + brownian_force
                # particle.position += particle.velocity
                pass
                
                # Apply boundary constraints to keep particles in view
                for i in range(3):
                    if abs(particle.position[i]) > 0.95:
                        particle.velocity[i] *= -0.5
                        particle.position[i] = np.clip(particle.position[i], -0.95, 0.95)
    
    def draw_text(self, text, position, color=(255, 255, 255), large=False):
        font = self.large_font if large else self.font
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_width, text_height = text_surface.get_size()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glWindowPos2d(position[0], position[1])
        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera transformations
        glTranslatef(0.0, 0.0, -self.camera_distance)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        # Enable lighting for 3D objects
        glEnable(GL_LIGHTING)
        
        # Draw biological tissue environment - semi-transparent box to represent boundaries
        glPushMatrix()
        glColor4f(0.9, 0.9, 0.95, 0.1)  # Very light, nearly transparent
        self.draw_wireframe_cube(1.0)
        glPopMatrix()
        
        # Draw cells in the tissue
        for pos, size, color in self.tissue.cells:
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            glColor4f(*color)
            self.draw_sphere(size, 12, 12)
            glPopMatrix()
            
        # Draw blood vessels in the tissue
        for start, end, radius, color in self.tissue.vessels:
            glColor4f(*color)
            self.draw_cylinder(start, end, radius, 8)
            
        # Draw ECM particles
        for pos, size, color in self.tissue.ecm_particles:
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            glColor4f(*color)
            self.draw_sphere(size, 6, 6)
            glPopMatrix()
        
        # Draw target regions
        for target in self.targets:
            glPushMatrix()
            glTranslatef(target.position[0], target.position[1], target.position[2])
            pulse_size = target.size * (1.0 + 0.1 * math.sin(target.pulse_factor * 2 * math.pi))
            glColor4f(*target.color)
            self.draw_sphere(pulse_size, 16, 16)
            glPopMatrix()
        
        # Draw particles
        for particle in self.particles:
            glPushMatrix()
            glTranslatef(particle.position[0], particle.position[1], particle.position[2])
            # Make grabbed particles brighter
            if particle.is_grabbed:
                color = np.array(particle.color) * 1.5
                color = np.clip(color, 0, 1)
                glColor3f(*color)
            else:
                glColor3f(*particle.color)
            self.draw_sphere(particle.size, 12, 12)
            glPopMatrix()
        
        # Disable lighting for laser beam
        glDisable(GL_LIGHTING)
        
        # Draw laser pointer
        glPushMatrix()
        glTranslatef(self.laser.position[0], self.laser.position[1], self.laser.position[2])
        glColor3f(*self.laser.color)
        self.draw_sphere(self.laser.size, 12, 12)
        glPopMatrix()
        
        # Draw laser beam (line from outside to laser position)
        glBegin(GL_LINES)
        glColor4f(*self.laser.color, 0.2)  # Faded start
        beam_origin = np.array([self.laser.position[0]*2, self.laser.position[1]*2, -2.0])
        glVertex3f(*beam_origin)
        
        glColor4f(*self.laser.color, self.laser.beam_intensity)  # Full intensity at pointer
        glVertex3f(*self.laser.position)
        glEnd()
        
        # Draw a wider laser beam using GL_TRIANGLE_STRIP for better visualization
        beam_width = self.laser.beam_width
        beam_dir = self.laser.position - beam_origin
        beam_len = np.linalg.norm(beam_dir)
        if beam_len > 0:
            beam_dir = beam_dir / beam_len
            # Calculate perpendicular vectors
            if abs(beam_dir[1]) < 0.9:
                perp1 = np.cross(beam_dir, [0, 1, 0])
            else:
                perp1 = np.cross(beam_dir, [1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(beam_dir, perp1)
            
            glBegin(GL_TRIANGLE_STRIP)
            for i in range(9):
                angle = i * math.pi / 4
                # Calculate point on circumference
                offset = perp1 * math.cos(angle) + perp2 * math.sin(angle)
                
                # Origin point (thinner)
                glColor4f(*self.laser.color, 0.05)
                glVertex3f(*(beam_origin + offset * beam_width * 0.5))
                
                # Endpoint (wider)
                glColor4f(*self.laser.color, self.laser.beam_intensity * 0.8)
                glVertex3f(*(self.laser.position + offset * beam_width))
            
            # Close the strip
            angle = 0
            offset = perp1 * math.cos(angle) + perp2 * math.sin(angle)
            glColor4f(*self.laser.color, 0.05)
            glVertex3f(*(beam_origin + offset * beam_width * 0.5))
            glColor4f(*self.laser.color, self.laser.beam_intensity * 0.8)
            glVertex3f(*(self.laser.position + offset * beam_width))
            glEnd()
        
        # Enable lighting again
        glEnable(GL_LIGHTING)
        
        # Disable lighting for text
        glDisable(GL_LIGHTING)
        
        # Draw UI elements
        self.draw_text(f"Particles Delivered: {self.particles_delivered}", (10, self.display_height - 30))
        
        # Display status info
        if self.info_mode > 0:
            y_pos = self.display_height - 60
            self.draw_text(f"Laser Position: ({self.laser.position[0]:.2f}, {self.laser.position[1]:.2f}, {self.laser.position[2]:.2f})", 
                          (10, y_pos))
            y_pos -= 25
            
            self.draw_text(f"Particles in Environment: {len(self.particles)}", (10, y_pos))
            y_pos -= 25
            
            # Show status of grabbed particle
            if self.laser.grabbed_particle:
                self.draw_text("Status: Particle grabbed", (10, y_pos), color=(0, 255, 0))
            else:
                self.draw_text("Status: No particle grabbed", (10, y_pos))
            y_pos -= 25
            
            if self.paused:
                self.draw_text("SIMULATION PAUSED", (self.display_width//2 - 100, self.display_height - 30), 
                              color=(255, 255, 0), large=True)
        
        # Display educational information in info mode 2
        if self.info_mode == 2:
            text = self.educational_text[self.current_info_text]
            self.draw_text(text, (self.display_width//2 - len(text)*4, 40), color=(255, 220, 150), large=True)
        
        # Show help if enabled
        if self.show_help:
            help_y = 120
            help_x = 10
            
            self.draw_text("Controls:", (help_x, help_y), color=(200, 200, 255))
            help_y -= 25
            self.draw_text("Arrow keys / WASD: Move laser", (help_x, help_y))
            help_y -= 20
            self.draw_text("Z/X: Move laser in/out", (help_x, help_y))
            help_y -= 20
            self.draw_text("G: Grab particle  R: Release particle", (help_x, help_y))
            help_y -= 20
            self.draw_text("Right mouse + drag: Rotate view", (help_x, help_y))
            help_y -= 20
            self.draw_text("Mouse wheel: Zoom in/out", (help_x, help_y))
            help_y -= 20
            self.draw_text("P: Pause/Resume  H: Hide/Show this help", (help_x, help_y))
            help_y -= 20
            self.draw_text("I: Toggle info display  1: Reset view", (help_x, help_y))
        
        pygame.display.flip()
    
    def draw_sphere(self, radius, slices, stacks):
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
    
    def draw_wireframe_cube(self, size):
        half_size = size / 2
        
        # Draw wireframe cube
        glBegin(GL_LINES)
        
        # Front face
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        # Back face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(-half_size, -half_size, -half_size)
        
        # Connecting lines
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, -half_size)
        
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glEnd()
    
    def draw_cube(self, size):
        half_size = size / 2
        
        # Draw a cube with given size
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glEnd()

    def draw_cylinder(self, start, end, radius, segments):
        # Create a cylinder from start to end points
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        
        # Calculate cylinder direction
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 0.0001:  # Avoid division by zero
            return
            
        direction = direction / length
        
        # Find rotation axis and angle
        if abs(direction[2] - 1.0) < 0.0001:
            # Special case: already aligned with z-axis
            rotation_axis = [0, 1, 0]
            rotation_angle = 0
        elif abs(direction[2] + 1.0) < 0.0001:
            # Special case: pointing negative z-axis
            rotation_axis = [0, 1, 0]
            rotation_angle = 180
        else:
            # General case
            z_axis = np.array([0, 0, 1], dtype=np.float32)
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, direction)) * 180 / np.pi
        
        # Draw cylinder
        glPushMatrix()
        glTranslatef(start[0], start[1], start[2])
        glRotatef(rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        
        # Create quadric for cylinder
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, radius, radius, length, segments, 1)
        
        # Draw end caps (optional)
        gluDisk(quadric, 0, radius, segments, 1)
        glTranslatef(0, 0, length)
        gluDisk(quadric, 0, radius, segments, 1)
        
        gluDeleteQuadric(quadric)
        glPopMatrix()
        
    def run(self):
        while self.running:
            self.handle_events()
            
            # Calculate time delta
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            self.update(dt)
            self.draw_scene()
            
            # Cap the frame rate
            pygame.time.wait(10)
        
        # Clean up
        if self.laser_connected:
            try:
                self.s.close()
            except:
                pass
                
        pygame.quit()
        
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()