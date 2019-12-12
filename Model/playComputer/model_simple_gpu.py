''' Othello AI, a project to create a naive Othello 
playing algorithm and evaluate its effectiveness 
Written by Colin Snow and Meg Ku
Structure adapted from Pytorch reinforcement q learning tutorial:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

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

import crunner

env = crunner.Move() #Othello game

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''
    Stores (state, action, next state, and reward) for every move that has ever been taken
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def update_memory(self, score, num_iterations):
        '''
        changes reward value based on tree search
        '''
        print('Updating Memory')
        for i in range (num_iterations):
            
            # set the new reward based on the iteration and existing reward
            new_reward = np.array([self.memory[len(self.memory)-1-i].reward.data.cpu().numpy().flatten()])[0] + score
            self.memory[len(self.memory)-1-i] = Transition(self.memory[len(self.memory)-1-i].state, \
                self.memory[len(self.memory)-1-i].action, self.memory[len(self.memory)-1-i].next_state,\
                torch.tensor([[new_reward]], device=device, dtype=torch.long))

class DQN(nn.Module):
    ''' Our neural network class which takes in a board and predicts 
    the expected reward for each move that it could take '''

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # We did a lot of experimentation with model sizes and types, so we decided to leave some
        # of our attempts here
        #1
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        #2
        # # Number of Linear input connections depends on output of conv2d layers
        # # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32


        #3
        # self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.fcl1 = nn.Linear(12544,5000)
        # self.fcl2 = nn.Linear(5000, 64)

        #4
        # Our final network choice. It is wide enough to capture a lot of detail but not too
        # large to have problems with vanishing gradients on such a small sample size
        self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.fcl1 = nn.Linear(20736,10000)
        self.fcl2 = nn.Linear(10000, 64)

        #5
        # Big model
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 256, kernel_size=2, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.fcl1 = nn.Linear(30976,10000)
        # self.fcl2 = nn.Linear(10000, 1000)
        # self.fcl3 = nn.Linear(1000,64)

        #6
        # #Medium model. - won 43% (once)
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # self.bn1   = nn.BatchNorm2d(16)
        # # self.conv2 = nn.Conv2d(16, 256, kernel_size=3, stride=1, padding=1)
        # # self.bn2   = nn.BatchNorm2d(256)
        # self.fc1   = nn.Linear(1024, 5        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = x.view(-1, 12544)
        # x = F.relu(self.fcl1(x))
        # x = F.log_softmax(self.fcl2(x))12)
        # self.fc2   = nn.Linear(512, 256)
        # self.fc3   = nn.Linear(256, 64)

        #7
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.fcl1 = nn.Linear(5184,2000)
        # self.fcl2 = nn.Linear(2000, 64)

        #8
        # Puny
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # # self.fcl1 = nn.Linear(1296,64)

        # self.fcl1 = nn.Linear(1296,500)
        # self.fcl2 = nn.Linear(500, 64)

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

BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = .9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 20

# Gets the inital game state
init_screen = get_screen()
screen_height, screen_width = 8,8 # Board is 8x8 squares

#Specifies the total number of actions that can be taken
n_actions = 64

#Initialize two networks, one to continuously train and one to select actions
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#Define optimizer and learnign rate
# We chose Adam because it adaptively chooses learning rates
# and determined our intial rate through extensive experimentation
optimizer = optim.Adam(policy_net.parameters(), lr= 1e-10)

#Create a memory object which stores the games played
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state, env):
    """Model is selecting action based on max of available.
    we want it to select the max of the board, and not proceed until it chooses one of the available ones """

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # Decide whether to use the model or a ransom choice to exlore
    # The more iterations that are done, the more it relies on the network to choose
    if sample > eps_threshold:
        with torch.no_grad():
            
            # Evaluate the network for the current state
            policynet = policy_net(state)

            # Get a list of legal moves for the current player
            possibleMoves = env.state()[2]

            # Mask the probabilities returned by the network to only include legal moves
            possiblepolicy = torch.tensor([policynet[0][i] for i in possibleMoves])
            policynet = np.array([policynet.data.cpu().numpy().flatten()])[0]
            policynet = [policynet[i] if i in possibleMoves else 0 for i in range(64)]
            policynet = np.array(policynet)

            # Normalize policy so that it can be sampled probabilistically
            policynet = policynet / np.sum(policynet)

            if math.isnan(policynet[0]):
                # Occurs when the gradient vanishes (does not normally occur in our model)
                print('got nan')
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            else:
                # Sample a move probabilistically from the legal options
                policymax = np.random.choice(64, p=policynet)
            
            # Ensure the move is legal and return it
            if policymax in possibleMoves:
                return torch.tensor([[policymax]], device=device, dtype=torch.long)
            # Choose a random move if it is not
            else:
                print('made random choice')
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)

    else:
        # Return a random legal move
        return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)


def optimize_model():

    # If the data is too small do not start sampling
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
             if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.ones(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    LOSSES.append(loss.cpu().item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


############# THE GAME #############

LOSSES = []
num_episodes = 200 #num games
comp_scores = []
user_scores = []

# Iterate for each game
for i_episode in range(num_episodes):

    # Initialize the environment and state
    env.reset()
    last_screen = get_screen() # retrieve board in pytorch tensor
    current_screen = get_screen()
    state = current_screen

    for t in count(1): # Iterate through moves until terminated

        # Select and perform an action
        state = get_screen()
        action = select_action(state, env).to(device)
        _,_,_, score, finished = env.make_move(action.item())

        # Record reward (0 in this case because the game is not over)
        reward = 0
        reward = torch.tensor([reward], device=device)

        # Read the new state after the move is performed
        next_state = get_screen()

        #Store the current state, the chosen action, the next state and the reward for that state in memory
        memory.push(state, action, next_state, reward)

        # Reward time
        if finished:
            # Assign a +1 reward if it wins and a -1 if it loses
            if score[1] > score[0]:
                reward_add = 1
            else:
                reward_add = -1
            # Update the memory with this reward
            memory.update_memory(reward_add, t)
            print(score)
            #Save the scores of user and computer
            comp_scores.append(score[0])
            user_scores.append(score[1])
            break

    # Print episode number
    print('Episode ' + str(i_episode) + ' out of ' + str(num_episodes))

    # Perform a step of optimization
    optimize_model()

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print('Complete')

# Print statistics on how the games went
wins = sum([comp_scores[i] < user_scores[i] for i in range(len(comp_scores))])
user_average = statistics.mean(user_scores)
standard_deviation = statistics.stdev(user_scores)
vert = np.array(user_scores).reshape((-1, 1))
horiz = np.array(range(len(user_scores))).reshape((-1, 1))
model = LinearRegression().fit(horiz, vert)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print("AI won " + str(wins) + " games out of " + str(len(user_scores)))
print("AI average was " + str(user_average) )
print("AI stdev was " + str(standard_deviation) )

print(user_scores)

# Save Model
torch.save(policy_net.state_dict(), 'state_dict_final.pyt')

# Record loss data
print(LOSSES)









# Stuff below here is for analysis only!!! 
# This section plays the game against the final network to see how good it is.

def new_select_action(state, env):
    with torch.no_grad():
            policynet = policy_net(state)
            possibleMoves = env.state()[2]
            possiblepolicy = torch.tensor([policynet[0][i] for i in possibleMoves])
            policynet = np.array([policynet.data.cpu().numpy().flatten()])[0]
            policynet = [policynet[i] if i in possibleMoves else 0 for i in range(64)]
            policynet = np.array(policynet)
            policynet = np.square(policynet)
            policynet = policynet / np.sum(policynet)
            if math.isnan(policynet[0]):
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            else:
                policymax = np.random.choice(64, p=policynet)

            return torch.tensor([[policymax]], device=device, dtype=torch.long)

def new_evaluate_model(iterations):
    
    policy_net.load_state_dict(torch.load('state_dict_final.pyt'))
    policy_net.eval()

    num_episodes = 10
    comp_scores = []
    user_scores = []
    for i in range(iterations):

        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen

        for t in count():

            action = new_select_action(state, env).to(device)
            _,_,_, score, finished = env.make_move(action.item())
            reward = float(score[1])
            # print(reward)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            memory.push(state, action, next_state, reward)
            if finished:
                print(score)
                # print(env.state()[0])
                # if num_episodes -  i_episode <= 100:
                comp_scores.append(score[0])
                user_scores.append(score[1])
                break

    wins = sum([comp_scores[i] < user_scores[i] for i in range(len(comp_scores))])
    user_average = statistics.mean(user_scores)
    standard_deviation = statistics.stdev(user_scores)

    vert = np.array(user_scores).reshape((-1, 1))
    horiz = np.array(range(len(user_scores))).reshape((-1, 1))
    model = LinearRegression().fit(horiz, vert)

    print('intercept:', model.intercept_)
    print('slope:', model.coef_)


    print("AI won " + str(wins) + " games out of " + str(len(user_scores)))
    print("AI average was " + str(user_average) )
    print("AI stdev was " + str(standard_deviation) )

    print(user_scores)

new_evaluate_model(100)
