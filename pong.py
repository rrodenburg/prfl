import pygame,sys
import random
import numpy as np
from pygame.locals import *

pygame.init()
fps = pygame.time.Clock()

#colors
WHITE = (255,255,255)
BLACK = (0,0,0)

#globals
WIDTH = 600
HEIGTH = 400
BALL_RADIUS = 10
PAD_WIDTH = 20
PAD_HEIGTH = 80
VELOCITY = 8 #pad velocity
pad_velocity_ai = 1.5 #ai velocity of computer opponent
number_of_players = 1

DistPadWall = 20 #distance paddles <-> wall


#canvas declaration
window = pygame.display.set_mode((WIDTH,HEIGTH))
pygame.display.set_caption('PONGG')

#function to draw the current screen:
def draw(canvas,_padL_pos,_padR_pos,_ball_pos):
    pygame.draw.rect(canvas,WHITE,(_padL_pos[0],_padL_pos[1],PAD_WIDTH,PAD_HEIGTH)) #draws L pad
    pygame.draw.rect(canvas,WHITE,(_padR_pos[0],_padR_pos[1],PAD_WIDTH,PAD_HEIGTH)) #draws R pad
    pygame.draw.rect(canvas,WHITE,(_ball_pos[0],_ball_pos[1],2*BALL_RADIUS,2*BALL_RADIUS)) #draws ball
    return

#function that does stuff upon pushing a button
def keydown(event,_padL_vel,_padR_vel): #inputs old y-velocities of pads, outputs (y-velocity of L pad, y-velocity of R pad)
    if event.key == K_w:
        _padL_vel = -VELOCITY        
        print("L player PUSHES!!")
        print("L&R velocities are now ({},{}):".format(_padL_vel, _padR_vel))
    elif event.key == K_s:
        _padL_vel = VELOCITY        
        print("L player PUSHES!!")
        print("L&R velocities are now ({},{}):".format(_padL_vel, _padR_vel))
    elif event.key == K_UP:        
        _padR_vel = -VELOCITY    
        print("R player PUSHES!!")
        print("L&R velocities are now ({},{}):".format(_padL_vel, _padR_vel))
    elif event.key == K_DOWN:        
        _padR_vel = VELOCITY
        print("R player PUSHES!!")
        print("L&R velocities are now ({},{}):".format(_padL_vel, _padR_vel))

    return (_padL_vel,_padR_vel)

def keyup(event,_padL_vel,_padR_vel):#inputs old y-velocities of pads, outputs (y-velocity of L pad, y-velocity of R pad)
    #if R-player releases up or down, reset velocity to 0:
    if event.key == K_UP and _padR_vel == -VELOCITY:
        _padR_vel = 0                    
        print("R player RELEASES!!")
        print("L&R velocities are now ({},{})".format(_padL_vel, _padR_vel))
    if event.key == K_DOWN and _padR_vel == VELOCITY:         
        _padR_vel = 0            
        print("R player RELEASES!!")
        print("L&R velocities are now ({},{})".format(_padL_vel, _padR_vel))
    #if L-player releases w or s, reset velocity to 0:
    if event.key == K_w and _padL_vel == -VELOCITY:
        _padL_vel = 0
        print("L player RELEASES!!")
        print("L&R velocities are now ({},{})".format(_padL_vel, _padR_vel))
    if event.key == K_s and _padL_vel == VELOCITY:         
        _padL_vel = 0
        print("L player RELEASES!!")
        print("L&R velocities are now ({},{})".format(_padL_vel, _padR_vel))

    return (_padL_vel,_padR_vel)
#KNOWN BUG: if you push down "up", then push down "down", then release "down", you're standing still, and not moving up as you'd like cause only "up" is being pushed down. Don't know how to fix this right now.

def updatepad_pos(velocity,currentypos): #returns new y-position of a pad
    if currentypos > 0 and currentypos < (HEIGTH - PAD_HEIGTH):
        return currentypos +  velocity
    elif currentypos == 0 and velocity > 0:
        return currentypos +  velocity
    elif currentypos == HEIGTH - PAD_HEIGTH and velocity < 0:
        return currentypos +  velocity
    else:
        return currentypos

def updateball_pos(velocity,currentpos,_padL_pos,_padR_pos): #returns new ball position; 
#the velocity-input of this function is the old velocity; we'll update it to the new velocity if necessary, namely in case of:
    #collision with upper wall or lower wall:
    if (currentpos[1] + velocity[1]) <= 0 or (currentpos[1] + 2*BALL_RADIUS) >= HEIGTH:        
        velocity[1] = -velocity[1]
    #collision with L pad
    if ((currentpos[0]+velocity[0]) <= (_padL_pos[0]+PAD_WIDTH)                        #x-coordinates of collision condition
    and (_padL_pos[1] + PAD_HEIGTH - 2*BALL_RADIUS) >= currentpos[1] >= _padL_pos[1]): #y-coordinates of collision conditionv
        velocity[0] = abs(velocity[0])
    #collision wit R pad
    if (((currentpos[0] + velocity[0] + 2 * BALL_RADIUS)  >= _padR_pos[0])              #x-coordinates of collision condition
    and (_padR_pos[1] + PAD_HEIGTH - 2*BALL_RADIUS) >= currentpos[1] >= _padR_pos[1]):  #y-coordinates of collision conditionv
        velocity[0] = -(abs(velocity[0]))
    newposition = currentpos + velocity
    return (velocity,newposition)
    
def wincheck(ball_position, ball_velocity, score): #Check if player wins
    if ball_position[0] <= 0:
        score[0] += 1
        print("player right wins, current score is {}".format(score))
        (ball_position, ball_velocity) = initialize_ball()
    if (ball_position[0] + 2 * BALL_RADIUS) >= WIDTH:
        score[1] += 1
        print('player left wins, current score is {}'.format(score))
        (ball_position, ball_velocity) = initialize_ball()
    return (score, ball_position, ball_velocity)
        
def initialize_ball():
    ball_pos = np.array([WIDTH/2-BALL_RADIUS,HEIGTH/2-BALL_RADIUS])
    ball_vel = np.array([random.choice(velinit_list),random.choice(velinit_list)])
    return (ball_pos, ball_vel)

def updatepad_pos_ai(velocity, currentypos, ball_position):
    if currentypos + 0.5*PAD_HEIGTH > ball_position[1]:
        return currentypos - velocity
    if currentypos + 0.5*PAD_HEIGTH < ball_position[1]:
        return currentypos + velocity
    else:
        return currentypos

#initialize variables
score = [0,0]
padL_vel = 0 #y-velocity of left pad
padR_vel = 0 #y-velocity of right pad
padL_pos= np.array([DistPadWall,HEIGTH/2-PAD_HEIGTH/2]) #these positions denote the upperleft corner of the pad; (0,0) is UL corner of screen
padR_pos= np.array([WIDTH - DistPadWall - PAD_WIDTH, HEIGTH/2-PAD_HEIGTH/2])

velinit_list = list(range(3,5)) + list(range(-4,-2))

#main game loop
running = True
(ball_pos, ball_vel) = initialize_ball()
while running:
    (ball_vel, ball_pos) = updateball_pos(ball_vel,ball_pos,padL_pos,padR_pos)
    padL_pos[1] = updatepad_pos(padL_vel,padL_pos[1])
    if number_of_players == 2:
        padR_pos[1] = updatepad_pos(padR_vel,padR_pos[1])
    else:
        padR_pos[1] = updatepad_pos_ai(pad_velocity_ai, padR_pos[1], ball_pos)
    window.fill(BLACK)
    draw(window, padL_pos, padR_pos,ball_pos)
    (score, ball_pos, ball_vel) = wincheck(ball_pos, ball_vel, score)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
            running = False
        if event.type == KEYDOWN: #keydown means you are pushing any button
            (padL_vel,padR_vel) = keydown(event,padL_vel,padR_vel)
        if event.type == KEYUP:
            (padL_vel,padR_vel) = keyup(event,padL_vel,padR_vel)
    pygame.display.update()
    fps.tick(60)
