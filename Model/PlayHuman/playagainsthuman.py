''' Implementation of our model to allow humans to play against the network.
Running this file will create an interactive command-line game which faces
you against our trained model '''
import sys
sys.path.append(".")

# Bad but works
import warnings
warnings.filterwarnings("ignore")


import statistics
from sklearn.linear_model import LinearRegression

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import playhuman

env = playhuman.Move() #Othello game


class bcolors:
    ''' Use command-line colors'''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    ''' Define size of neural net '''
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.fcl1 = nn.Linear(20736,10000)
        self.fcl2 = nn.Linear(10000, 64)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 20736)
        x = F.relu(self.fcl1(x))
        x = F.log_softmax(self.fcl2(x))

        return x

def get_screen():
    """Returns the current board in an 8 by 8 pytorch tensor"""

    screen = env.state()[0]
    screen = np.expand_dims(screen, axis=0)
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).to(device)   # Removed resize because screen is already small


env.reset()


# Gets the inital game state
init_screen = get_screen()
screen_height, screen_width = 8,8 # Board is 8x8 squares

#Specifies the total number of actions that can be taken
n_actions = 64

#Initialize two networks, one to continuously train and one to select actions
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load('testmodel.pyt', map_location={'cuda:0': 'cpu'}))

#Define optimizer and learnign rate
optimizer = optim.Adam(policy_net.parameters(), lr= 1e-3)

steps_done = 0

def select_action(state, env):
    """Model is selecting action based on max of available.
    we want it to select the max of the board, and not proceed until it chooses one of the available ones """

    with torch.no_grad():

        policynet = policy_net(state)
        possibleMoves = env.state()[2]

        possiblepolicy = torch.tensor([policynet[0][i] for i in possibleMoves])

        policynet = np.array([policynet.data.cpu().numpy().flatten()])[0]
        policynet = [policynet[i] if i in possibleMoves else 0 for i in range(64)]
        policynet = np.array(policynet)
        policynet = policynet / np.sum(policynet)

        if math.isnan(policynet[0]):
            print('got nan')
            return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
        else:
            policymax = np.random.choice(64, p=policynet)
        
        if policymax in possibleMoves:
            return torch.tensor([[policymax]], device=device, dtype=torch.long)

        else:
            print('made random choice')
            return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)


##### THE GAME #####

notcomputer = False
notplayer = False

print("OTHELLO AI by Colin Snow and Meg Ku")
print("You are currently playing a model which was trained to play the game by observing and playing games against a traditionan game-tree style opponent")
print('You are the 0 player and can choose any of the moves listed above the board')
while 1:

    state = get_screen()
    if env.state()[2] is not None and env.state()[2] != []:
        action = select_action(state, env).to(device)
        _,_,_,score,done = env.make_move(action.item(), 'computer')
        notcomputer = False
    else:
        print('AI can not move')
        env.make_move(0, 'computer')
        notcomputer = True

    state = get_screen()
    if env.state()[2] is not None and env.state()[2] != []:
        print(bcolors.BOLD + 'Moves:' + str(env.state()[2]) + bcolors.ENDC)
        board = state.cpu().numpy()
        board = list(board[0][0].flatten())
        symbolicboard = ['@ ' if i == 1 else '0 ' if i==-1 else '  ' for i in board ]
        symbolicboard = [str(i) if i in env.state()[2] else symbolicboard[int(i)] for i in range(0,64) ]

        nboard = np.array([symbolicboard])
        nboard = np.reshape(nboard,(8,8))
        print(nboard)

        while 1:
            x = input(bcolors.OKGREEN + "Your move? " + bcolors.ENDC)
            x = int(x) if x is not '' else -1
            if x in env.state()[2]:
                break
            else:
                print(bcolors.FAIL + 'Invalid move' + bcolors.ENDC)

        _,_,_,_,_ = env.make_move(x, 'player')
        notplayer = False
    else:
        print('you could not move')
        env.make_move(0, 'player')
        notplayer = True

    if done or (notcomputer and notplayer):
        print(score)
        if score[1] > score[0]:
            print(bcolors.OKGREEN + "Congradulations! you won" + bcolors.ENDC)
        else:
            print(bcolors.FAIL + "Sorry, you lost" + bcolors.FAIL)
        break
