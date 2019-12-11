
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
    Stores (state, action, next state, and reward) for each board setup
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
            
            new_reward = np.array([self.memory[len(self.memory)-1-i].reward.data.cpu().numpy().flatten()])[0] + score
            self.memory[len(self.memory)-1-i] = Transition(self.memory[len(self.memory)-1-i].state,self.memory[len(self.memory)-1-i].action, self.memory[len(self.memory)-1-i].next_state, torch.tensor([[new_reward]], device=device, dtype=torch.long))

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # # Number of Linear input connections depends on output of conv2d layers
        # # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32



        # self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.fcl1 = nn.Linear(12544,10000)
        # self.fcl2 = nn.Linear(10000, 64)

        # self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.fcl1 = nn.Linear(20736,10000)
        # self.fcl2 = nn.Linear(10000, 64)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 256, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fcl1 = nn.Linear(30976,10000)
        self.fcl2 = nn.Linear(10000, 1000)
        self.fcl3 = nn.Linear(1000,64)






    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = x.view(-1, 12544)
        # x = F.relu(self.fcl1(x))
        # x = F.relu(self.fcl2(x))

        # x = F.relu(self.bn1(self.conv1(x)))
        # # x = F.relu(self.conv2(x))
        # x = x.view(-1, 20736)
        # x = F.relu(self.fcl1(x))
        # x = F.relu(self.fcl2(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 30976)
        # print(x)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = F.relu(self.fcl3(x))

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




# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10

BATCH_SIZE = 256
GAMMA = 0.999 #TODO: understand this
EPS_START = .9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

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
optimizer = optim.Adam(policy_net.parameters(), lr=.001)

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
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policynet = policy_net(state)
            # print(policynet)
            possibleMoves = env.state()[2]

            possiblepolicy = torch.tensor([policynet[0][i] for i in possibleMoves])

            policynet = np.array([policynet.data.cpu().numpy().flatten()])[0]
            policynet = [policynet[i] if i in possibleMoves else .00001 for i in range(64)]
            policynet = np.array(policynet)
            policynet = policynet / np.sum(policynet)

            if math.isnan(policynet[0]):
                print('got nan')
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            else:
                policymax = np.random.choice(64, p=policynet)
                # policymax = np.argmax(policynet)
                # print(policymax)
            
            if policymax in possibleMoves:
                return torch.tensor([[policymax]], device=device, dtype=torch.long)

            else:
                print('made random choice')
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)

    else:
        # print("random")
        return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    pass


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # print('Memory length: ' + str(len(memory)))
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    # print(batch.state)
    # print(batch.state.size())
    non_final_next_states = torch.cat([s for s in batch.next_state
                                               # print(env.state()[0])
             if s is not None])
    # print(non_final_next_states)
    state_batch = torch.cat(batch.state)
    # print(batch.action)
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

    # print('rewards:')
    # print(reward_batch)
    # print('sav:')
    # print(state_action_values)
    # print('esav:')
    # print(expected_state_action_values)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    LOSSES.append(loss.cpu().item())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



##### THE GAME #####

LOSSES = []
num_episodes = 100 #num games
comp_scores = []
user_scores = []

# Gameplay
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen() # retrieve board in pytorch tensor
    current_screen = get_screen()
    state = current_screen
    # print(state)
    for t in count(1):
        # Select and perform an action
        state = get_screen()
        action = select_action(state, env).to(device)
        _,_,_, score, finished = env.make_move(action.item())
        reward = float(score[1])
        # print(reward)
        reward = torch.tensor([reward], device=device)


        next_state = get_screen()

        #Story the current state, the chosen action, the next state and the reward for that state in memory
        memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the target network)
        
        # Reward time
        if finished:
            # reward_add = int(score[1] > score[0]) * 100
            if score[1] > score[0]:
                reward_add = 10000
            else:
                reward_add = 0
            memory.update_memory(reward_add, t)
            print(score)
            episode_durations.append(reward)
            plot_durations()
            # print(env.state()[0])
            # if num_episodes -  i_episode <= 100:
            
            #Save the scores of user and computer
            comp_scores.append(score[0])
            user_scores.append(score[1])
            break

    print('Episode ' + str(i_episode) + ' out of ' + str(num_episodes))
    optimize_model()
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())




print('Complete')

torch.save(policy_net.state_dict(), 'state_dict_final.pyt')


print(LOSSES)

print(memory.sample(10))










# Stuff velow here is for analysis only!!!







def new_select_action(state, env):
    with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policynet = policy_net(state)
            # print(policynet)
            possibleMoves = env.state()[2]
            # print(possibleMoves)
            # print(policynet[0])
            # testlist = [policynet[0][i].data.cpu().numpy() for i in possibleMoves]
            # print(testlist)
            possiblepolicy = torch.tensor([policynet[0][i] for i in possibleMoves])
            # print('possible policy is')
            # print(possiblepolicy)
            policynet = np.array([policynet.data.cpu().numpy().flatten()])[0]

            policynet = [policynet[i] if i in possibleMoves else 0 for i in range(64)]
            # print(policynet)
            policynet = np.array(policynet)
            policynet = np.square(policynet)
            policynet = policynet / np.sum(policynet)
            # print(policynet)
            if math.isnan(policynet[0]):
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            else:
                policymax = np.random.choice(64, p=policynet)
                # policymax = np.argmax(policynet)
                # print(policymax)

            # return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            return torch.tensor([[policymax]], device=device, dtype=torch.long)


        # return policymax

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
                episode_durations.append(reward)
                plot_durations()
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




# def s(iterations):



# plt.plot(horiz, vert)
# plt.show()





