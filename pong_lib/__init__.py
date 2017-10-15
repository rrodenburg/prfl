import pygame,sys
import random
import numpy as np
from pygame.locals import *

class Pong(object):
    
    #colors
    white = (255,255,255)
    black = (0,0,0)
    
    def __init__(
                self, 
                number_of_players = 1, 
                width = 600, 
                heigth = 400, 
                ball_radius = 10, 
                pad_width = 20, 
                pad_heigth = 80, 
                pad_velocity = 8, 
                pad_velocity_ai = 1,
                DistPadWall = 20
                ):
                
        self.number_of_players = number_of_players
        self.width = width
        self.heigth = heigth
        self.ball_radius = ball_radius
        self.pad_width = pad_width
        self.pad_heigth = pad_heigth
        self.pad_velocity = pad_velocity
        self.pad_velocity_ai = pad_velocity_ai
        self.DistPadWall = DistPadWall
        
        #canvas declaration
        self.window = pygame.display.set_mode((self.width , self.heigth))
        
        #initialize variables for game
        self.score = [0,0]
        self.padL_vel = 0 #y-velocity of left pad
        self.padR_vel = 0 #y-velocity of right pad
        self.padL_pos= np.array([DistPadWall, self.heigth/ 2 - self.pad_heigth / 2]) #these positions denote the upperleft corner of the pad; (0,0) is UL corner of screen
        self.padR_pos= np.array([self.width - DistPadWall - self.pad_width, self.heigth / 2 - self.pad_heigth / 2])
        
    #function to draw the current screen:
    def draw(self, canvas, _padL_pos, _padR_pos, _ball_pos):
        pygame.draw.rect(canvas, Pong.white, (_padL_pos[0],_padL_pos[1], self.pad_width, self.pad_heigth)) #draws L pad
        pygame.draw.rect(canvas, Pong.white, (_padR_pos[0],_padR_pos[1], self.pad_width, self.pad_heigth)) #draws R pad
        pygame.draw.rect(canvas, Pong.white, (_ball_pos[0],_ball_pos[1], 2*self.ball_radius, 2*self.ball_radius)) #draws ball
        return
        
    #function that does stuff upon pushing a button
    def keydown(self, event, _padL_vel, _padR_vel): #inputs old y-velocities of pads, outputs (y-velocity of L pad, y-velocity of R pad)
        if event.key == K_w:
            _padL_vel = -self.pad_velocity        
        elif event.key == K_s:
            _padL_vel = self.pad_velocity        
        elif event.key == K_UP:        
            _padR_vel = -self.pad_velocity    
        elif event.key == K_DOWN:        
            _padR_vel = self.pad_velocity
        return (_padL_vel,_padR_vel)
        
    def keyup(self, event,_padL_vel,_padR_vel):#inputs old y-velocities of pads, outputs (y-velocity of L pad, y-velocity of R pad)
        #if R-player releases up or down, reset velocity to 0:
        if event.key == K_UP and _padR_vel == -self.pad_velocity:
            _padR_vel = 0                    
        if event.key == K_DOWN and _padR_vel == self.pad_velocity:         
            _padR_vel = 0            
        #if L-player releases w or s, reset velocity to 0:
        if event.key == K_w and _padL_vel == -self.pad_velocity:
            _padL_vel = 0
        if event.key == K_s and _padL_vel == self.pad_velocity:         
            _padL_vel = 0
        return (_padL_vel,_padR_vel)
    #KNOWN BUG: if you push down "up", then push down "down", then release "down", you're standing still, and not moving up as you'd like cause only "up" is being pushed down. Don't know how to fix this right now.  
    
    def updatepad_pos(self, velocity, currentypos): #returns new y-position of a pad
        if currentypos > 0 and currentypos < (self.heigth - self.pad_heigth):
            return currentypos +  velocity
        elif currentypos == 0 and velocity > 0:
            return currentypos +  velocity
        elif currentypos == (self.heigth - self.pad_heigth) and velocity < 0:
            return currentypos +  velocity
        else:
            return currentypos
            
    def updateball_pos(self, velocity,currentpos,_padL_pos,_padR_pos): #returns new ball position; 
    #the velocity-input of this function is the old velocity; we'll update it to the new velocity if necessary, namely in case of:
        #collision with upper wall or lower wall:
        if (currentpos[1] + velocity[1]) <= 0 or (currentpos[1] + 2 * self.ball_radius) >= self.heigth:        
            velocity[1] = -velocity[1]
        #collision with L pad
        if ((currentpos[0] + velocity[0]) <= (_padL_pos[0] + self.pad_width)                        #x-coordinates of collision condition
        and (_padL_pos[1] + self.pad_heigth - 2 * self.ball_radius) >= currentpos[1] >= _padL_pos[1]): #y-coordinates of collision conditionv
            velocity[0] = abs(velocity[0])
        #collision wit R pad
        if (((currentpos[0] + velocity[0] + 2 * self.ball_radius)  >= _padR_pos[0])              #x-coordinates of collision condition
        and (_padR_pos[1] + self.pad_heigth - 2 * self.ball_radius) >= currentpos[1] >= _padR_pos[1]):  #y-coordinates of collision conditionv
            velocity[0] = -(abs(velocity[0]))
        newposition = currentpos + velocity
        return (velocity,newposition)
        
    def wincheck(self, ball_position, ball_velocity, score): #Check if player wins
        if ball_position[0] <= 0:
            score[0] += 1
            print("player right wins, current score is {}".format(score))
            (ball_position, ball_velocity) = self.initialize_ball()
        if (ball_position[0] + 2 * self.ball_radius) >= self.width:
            score[1] += 1
            print('player left wins, current score is {}'.format(score))
            (ball_position, ball_velocity) = self.initialize_ball()
        return (score, ball_position, ball_velocity)
        
    def initialize_ball(self):
        velinit_list = list(range(3,5)) + list(range(-4,-2))
        ball_pos = np.array([self.width / 2 - self.ball_radius, self.heigth / 2 - self.ball_radius])
        ball_vel = np.array([random.choice(velinit_list),random.choice(velinit_list)])
        return (ball_pos, ball_vel)
        
    def updatepad_pos_ai(self, velocity, currentypos, ball_position):
        if currentypos + 0.5 * self.pad_heigth > ball_position[1]:
            return currentypos - velocity
        if currentypos + 0.5 * self.pad_heigth < ball_position[1]:
            return currentypos + velocity
        else:
            return currentypos
            
    def start(self, frames_per_sec = 60):
        #main game loop
        pygame.display.set_caption('PONG')
        self.fps = pygame.time.Clock()
        self.running = True
        (self.ball_pos, self.ball_vel) = self.initialize_ball()
        while self.running:
            (self.ball_vel, self.ball_pos) = self.updateball_pos(self.ball_vel, self.ball_pos, self.padL_pos, self.padR_pos)
            self.padL_pos[1] = self.updatepad_pos(self.padL_vel,self.padL_pos[1])
            if self.number_of_players == 2:
                self.padR_pos[1] = self.updatepad_pos(self.padR_vel,self.padR_pos[1])
            else:
                self.padR_pos[1] = self.updatepad_pos_ai(self.pad_velocity_ai, self.padR_pos[1], self.ball_pos)
            self.window.fill(Pong.black)
            self.draw(self.window, self.padL_pos, self.padR_pos, self.ball_pos)
            (self.score, self.ball_pos, self.ball_vel) = self.wincheck(self.ball_pos, self.ball_vel, self.score)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                    self.running = False
                if event.type == KEYDOWN: #keydown means you are pushing any button
                    (self.padL_vel, self.padR_vel) = self.keydown(event, self.padL_vel, self.padR_vel)
                if event.type == KEYUP:
                    (self.padL_vel,self.padR_vel) = self.keyup(event, self.padL_vel, self.padR_vel)
            pygame.display.update()
            self.fps.tick(frames_per_sec)
            
