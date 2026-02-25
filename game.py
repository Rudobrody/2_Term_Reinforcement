import pygame
import torch
import pygame
import numpy as np
import torch
import torch.nn as nn
import os 
import math
from abstract_car import AbstractCar
from utils import scale_image
from itertools import permutations
import numpy as np

from model_v1 import RacingDQN, PlayerCar2
from model_v2 import PlayerCarModel2
from dqn_car import DQNCar
from szmajda import NeuralPlayerCar
from dqn_agent import DQNAgent
#from dqn_32 import RacingDQN

#Based on https://github.com/techwithtim/Pygame-Car-Racer

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.35)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.35)
PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.35)
GRAY_CAR = scale_image(pygame.image.load("imgs/grey-car.png"), 0.35)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

pygame.font.init()  # Initialize the font module
FONT = pygame.font.Font(None, 24)  # Use a default font with size 24


FPS = 60

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=100):
    checkpoints = []
    for i in range(len(track_path) - 1):
        x1, y1 = track_path[i]
        x2, y2 = track_path[i + 1]
        for t in np.linspace(0, 1, num_checkpoints // len(track_path)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            checkpoints.append((int(x), int(y)))
    return checkpoints


CHECKPOINTS = generate_checkpoints(track_path)

def draw_checkpoints(win, checkpoints):
    for x, y in checkpoints:
        pygame.draw.circle(win, (0, 255, 0), (x, y), 5)

# In the game loop


class Game:
    def __init__(self, width, height, fps=60):
        self.win = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Racing Game")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.cars = []  # List to hold car objects
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
          (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        self.running = True

    def add_car(self, car):
        """Add a car to the game."""
        if not isinstance(car, AbstractCar):
            raise ValueError("Only instances of AbstractCar or its subclasses can be added.")

        if len(self.cars) == 0:
            car.set_image(RED_CAR)
            car.set_position((180, 200))
        elif len(self.cars) == 1:
            car.set_image(GREEN_CAR)
            car.set_position((150, 200))
        if len(self.cars) == 2:
            car.set_image(GRAY_CAR)
            car.set_position((180, 160))
        elif len(self.cars) == 3:
            car.set_image(PURPLE_CAR)
            car.set_position((150, 160))

        car.reset()
        self.cars.append(car)

    def draw(self):
        """Draw the background and all cars."""
        for img, pos in self.images:
            self.win.blit(img, pos)

        if not hasattr(self, 'name_font'):
            if not pygame.font.get_init():
                pygame.font.init()
            self.name_font = pygame.font.SysFont('Arial', 16, bold=True)

        for car in self.cars:
            car.draw(self.win)

            # 1. Pobieramy imię auta
            name_text = car.get_name()
            
            # 2. Tworzymy powierzchnię z tekstem (biały tekst)
            text_surface = self.name_font.render(name_text, True, (255, 255, 255))
            
            # 3. Dla lepszej czytelności (kontrast na trawie) robimy czarny obrys (cień)
            shadow_surface = self.name_font.render(name_text, True, (0, 0, 0))
            
            # 4. Obliczamy pozycję (środek auta, lekko nad nim)
            # Używamy self.x i self.y (górny lewy róg obrazka auta)
            car_center_x = car.x + car.img.get_width() / 2
            
            # Ustawiamy napis wyśrodkowany względem osi X auta i 15 pikseli nad autem
            text_x = car_center_x - text_surface.get_width() / 2
            text_y = car.y - 15 
            
            # 5. Rysujemy cień (przesunięty o 1 piksel)
            self.win.blit(shadow_surface, (text_x + 1, text_y + 1))
            
            # 6. Rysujemy właściwy biały tekst
            self.win.blit(text_surface, (text_x, text_y))
            # car.draw_rays(self.win, TRACK_BORDER_MASK)


        pygame.display.update()

    def check_collisions(self):

        for car in self.cars:
            if car.collide(TRACK_BORDER_MASK):
                car.bounce()

        """Check for collisions between cars."""
        for i, car1 in enumerate(self.cars):
            for j, car2 in enumerate(self.cars):
                if i != j and car1.collide_car(car2):
                    car1.bounce()
                    car2.bounce()
                    # print(f"Collision between Car {i+1} and Car {j+1}!")

    def check_finish_line(self):

        finished = []

        for car in self.cars:
            finish_poi_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
            if finish_poi_collide != None:
                if finish_poi_collide[1] == 0:
                    car.bounce()
                else:
                    finished.append(car.get_name())
                    self.cars.remove(car)

        return finished

    def move_cars(self):
        """Handle car movements."""

        for car in self.cars:
            car.update_progress(CHECKPOINTS)

        for car in self.cars:
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = car.get_distances_to_cars(self.cars)
            car.perform_action(car.choose_action([distances, car_distances, car.get_progress(), CHECKPOINTS]))

    def run(self):
        """Main game loop."""
        who_finished_first = []
        while self.running and len(self.cars) != 0:
            self.clock.tick(self.fps)
            draw_checkpoints(self.win, CHECKPOINTS) # to bylo zakomentowane
            pygame.display.update() # to bylo zakomentowane

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False



            self.move_cars()
            self.check_collisions()
            finish_lines = self.check_finish_line()
            if len(finish_lines) != 0:
                who_finished_first.append(finish_lines)

            self.draw()


        pygame.quit()
        print("Game over!")
        print(who_finished_first)
        return who_finished_first


class PlayerCar(AbstractCar):

    def __init__(self, name):
        # Call the AbstractCar __init__ method
        super().__init__(name)

    def choose_action(self, state):
        """
        Perform an action based on the input.

        Actions:
        - "forward": Move the car forward.
        - "backward": Move the car backward.
        - "left": Turn the car left.
        - "right": Turn the car right.
        - "stop": Reduce the car's speed.
        """

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            return "forward"
        elif keys[pygame.K_DOWN]:
            return "backward"
        elif keys[pygame.K_LEFT]:
            return "left"
        elif keys[pygame.K_RIGHT]:
            return "right"
        else:
            return "stop"


# class PlayerCar2(AbstractCar):

#     def __init__(self, name):
#         # Call the AbstractCar __init__ method
#         super().__init__(name)

#     def choose_action(self, state):
#         """
#         Determines the next action for the car based on the current state of the environment.

#         Parameters:
#             state (list): A 3-element list representing the car's current state:
#                 - state[0]: A list of 8 float values representing distances to the track border
#                             in 8 directions (every 45 degrees, starting from forward).
#                 - state[1]: A list of 8 float values representing distances to the nearest car
#                            in the same 8 directions.
#                 - state[2]: A 2-element list representing progress information:
#                             - state[2][0]: The index of the closest checkpoint.
#                             - state[2][1]: The car's progress, e.g., distance to the next checkpoint
#                                            or normalized progress value.

#         Returns:
#             - "forward": Move the car forward.
#             - "backward": Move the car backward.
#             - "left": Turn the car left.
#             - "right": Turn the car right.
#             - "stop": Reduce the car's speed.
#             """

#         """INSERT YOUR CODE HERE"""

#         keys = pygame.key.get_pressed()

#         if keys[pygame.K_w]:
#             return "forward"
#         elif keys[pygame.K_s]:
#             return "backward"
#         elif keys[pygame.K_a]:
#             return "left"
#         elif keys[pygame.K_d]:
#             return "right"
#         else:
#             return "stop"
# Path to the file with weights
#WEIGHTS_FILE = r'racing_agent_final.pth'
#WEIGHTS_FILE = r'racing_agent_ep10500.pth'
# WEIGHTS_FILE = r'grand_champion_model_v1.pth'

# class RacingDQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RacingDQN, self).__init__()
        
#         self.feature_layer = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
        
#         # Value Stream (V)
#         self.value_stream = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
        
#         # Advantage Stream (A)
#         self.advantage_stream = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_dim)
#         )

    
    # def forward(self, x):
    #     features = self.feature_layer(x)
    #     values = self.value_stream(features)
    #     advantages = self.advantage_stream(features)
    #     return values + (advantages - advantages.mean(dim=1, keepdim=True))

# class PlayerCar2(AbstractCar):
#     def __init__(self, name):
#         super().__init__(name)
        
#         # cpu for safety during tournament
#         self.device = torch.device("cpu") 
        
#         # Initialization
#         self.model = RacingDQN(input_dim=22, output_dim=5).to(self.device)

#         # Loading weights 
#         try:
#             script_dir = os.path.dirname(os.path.abspath(__file__))
#             weights_path = os.path.join(script_dir, WEIGHTS_FILE)
            
#             if os.path.exists(weights_path):
#                 self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
#                 self.model.eval()
#                 print(f"[{name}] Wagi zaladowane: {WEIGHTS_FILE}")
#             else:
#                 print(f"[{name}] Brak pliku wag: {WEIGHTS_FILE}")
#         except Exception as e:
#             print(f"[{name}] Blad ładowania wag: {e}")

#         # Defining actions
#         self.ACTIONS = ["forward", "backward", "left", "right", "stop"]
        

#     def choose_action(self, state):
#         """
#         Determines the next action for the car based on the current state of the environment.

#         Parameters:
#             state (list): A 3-element list representing the car's current state:
#                 - state[0]: A list of 8 float values representing distances to the track border
#                             in 8 directions (every 45 degrees, starting from forward).
#                 - state[1]: A list of 8 float values representing distances to the nearest car
#                            in the same 8 directions.
#                 - state[2]: A 2-element list representing progress information:
#                             - state[2][0]: The index of the closest checkpoint.
#                             - state[2][1]: The car's progress, e.g., distance to the next checkpoint
#                                            or normalized progress value.

#         Returns:
#             - "forward": Move the car forward.
#             - "backward": Move the car backward.
#             - "left": Turn the car left.
#             - "right": Turn the car right.
#             - "stop": Reduce the car's speed.
            
#         """
#         # If we didn't load a model, fallback to manual stearing
#         # if not hasattr(self, 'model'):
#         #     keys = pygame.key.get_pressed()
#         #     if keys[pygame.K_w]: return "forward"
#         #     if keys[pygame.K_s]: return "backward"
#         #     if keys[pygame.K_a]: return "left"
#         #     if keys[pygame.K_d]: return "right"
#         #     return "stop"

#         # Normalize as during the training
#         wall_dists = np.array(state[0]) / 1000.0 # 1000 because it was max distance
#         car_dists = np.array(state[1]) / 200.0 # 200 because it was of a range to detect a car

#         # Physics
#         velocity = self.vel / self.max_vel
        
#         # Calculateing the angle and distance to the next target
#         target_idx = int(state[2][0])
        
#         angle_to_target = 0.0
#         dist_to_target = 0.0

#         ###############################################################
#         # Added for the test
#         # car_cx = self.x + self.img.get_width() // 2
#         # car_cy = self.y + self.img.get_height() // 2
#         ##############################################################

#         if target_idx < len(CHECKPOINTS):
            
#             # Definine position of the next target
#             target_pos = CHECKPOINTS[target_idx]
            
#             dx = target_pos[0] - self.x
#             dy = target_pos[1] - self.y
#             ###################################################
#             # Added for the test
#             # dx = target_pos[0] - car_cx
#             # dy = target_pos[1] - car_cy
#             ####################################################

#             target_angle = math.degrees(math.atan2(-dx, -dy))

#             # Relative angle
#             diff_angle = (target_angle - self.angle) % 360

#             if diff_angle > 180: diff_angle -= 360
            
#             # Normalize -1 to 1
#             angle_to_target = diff_angle / 180.0
            
#             # Distance
#             dist_to_target = math.sqrt(dx**2 + dy**2) / 1000.0
#         else:
#             angle_to_target = 0
            
#         checkpoint_progress = target_idx / len(CHECKPOINTS)
#         car_angle = (self.angle % 360) / 360.0
        
#         # Build Tensor
#         state_vector = np.concatenate([
#             wall_dists,        # 8
#             car_dists,         # 8
#             [velocity],       # 1
#             [angle_to_target], # 1
#             [dist_to_target],  # 1
#             [checkpoint_progress],   # 1
#             [car_angle],  # 1
#             [1.0 if self.vel >= 0 else 0.0]  # 1
#         ])
#         # State tensor
#         state_tensor = torch.tensor(state_vector, dtype=torch.float32).to(self.device)
        
#         # Decision 
#         with torch.no_grad():
#             action_idx = self.model(state_tensor.unsqueeze(0)).argmax().item()
            
#         action_str = self.ACTIONS[action_idx]
#         print(f"Action chosen: {action_str}")
#         return action_str


def main():

    final_results = dict()

    #initializing players - it is possible to play up to 4 players together
    #players = [PlayerCar("P1"), PlayerCar2("P2"), PlayerCar("P1"), PlayerCar2("P2")]
    
    car = DQNCar("Artur", checkpoints=CHECKPOINTS, train=False)
    car.load ("dqn_model_best.pth")
    
    players = [PlayerCar2("Pawel"), car, NeuralPlayerCar("Szmajda", epsilon=0, weights_path="szmajda/racing_model.pth"), DQNAgent("Eryk", model_path='dqn_model_agent1_ep700.pth')]

    #players = [PlayerCar2("Pawel")]
    for p in players:
        final_results[p.get_name()] = 0

    perm = permutations(players)

    for p in perm:

        print(p)

        game = Game(WIDTH, HEIGHT, FPS)

        # Add cars
        for player in p:
            game.add_car(player)

        # Run the game
        temp_rank = game.run()

        points = len(players)

        for tr in temp_rank:
            for t in tr:
                final_results[t] += points
            points -= 1

    print(final_results)

if __name__ == "__main__":
    main()