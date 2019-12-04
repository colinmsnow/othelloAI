
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

# %matplotlib inline


env = crunner.Move() #Othello game


# env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 256, kernel_size=2, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.fcl1 = nn.Linear(30976,10000)
        # self.fcl2 = nn.Linear(10000, 1000)
        # self.fcl3 = nn.Linear(1000,64)
        self.fcl1 = nn.Linear(1296,10000)
        self.fcl2 = nn.Linear(10000, 64)



        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

        # x = x.view(-1, 30976)

        # x = F.relu(self.fcl1(x))
        # x = F.relu(self.fcl2(x))
        # x = F.relu(self.fcl3(x))
        
        x = x.view(-1, 1296)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))

        # return self.head(x.view(x.size(0), -1))
        # print(x)
        return x





# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])



# def get_cart_location(screen_width):
#     world_width = env.x_threshold * 2
#     scale = screen_width / world_width
#     return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # # Cart is in the lower half, so strip off the top and bottom of the screen
    # _, screen_height, screen_width = screen.shape
    # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # view_width = int(screen_width * 0.6)
    # cart_location = get_cart_location(screen_width)
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width // 2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width // 2,
    #                         cart_location + view_width // 2)
    # # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, slice_range]
    # # Convert to float, rescale, convert to torch tensor
    # # (this doesn't require a copy)



    screen = env.state()[0]
    screen = np.expand_dims(screen, axis=0)
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).to(device)   # Removed resize because screen is already small


env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()




# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.025
EPS_DECAY = 30
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
screen_height, screen_width = 8,8

# Get number of actions from gym action space
# n_actions = len(env.state()[2])
n_actions = 64

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


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
            policynet = policynet / np.sum(policynet)
            # print(policynet)
            if math.isnan(policynet[0]):
                print('got nan')
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            else:
                policymax = np.random.choice(64, p=policynet)
                # print(policymax)

            return torch.tensor([[policymax]], device=device, dtype=torch.long)
            # policymax =  possiblepolicy.max(1)[1]
            # _, index = possiblepolicy.max(0)
            # # print(value)
            # # print(index)
            # print(possibleMoves[index])
            # policymax = torch.tensor([[possibleMoves[index]]])
            # print(policymax)

            # if policymax.item() in possibleMoves:
            #     print("chosen")
            #     return torch.tensor([[policymax]])
            # else:
            #     # return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            #     print("Bad Move")
            #     reward = float(-1000)
            #     reward = torch.tensor([reward], device=device)
            #     memory.push(state, action, next_state, reward)
            #     optimize_model()
            #     target_net.load_state_dict(policy_net.state_dict())
        # return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)

    else:
        # print("random")
        return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    # plt.figure(2)
    # plt.clf()
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    # plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Score')
    # plt.plot(durations_t.numpy())
    # plt.show()
    # # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
    pass




def optimize_model():
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
    # print(batch.state)
    # print(batch.state.size())
    non_final_next_states = torch.cat([s for s in batch.next_state
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

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()




num_episodes = 100
comp_scores = []
user_scores = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen
    # print(state)
    for t in count():
        # Select and perform an action
        
        action = select_action(state, env).to(device)
        _,_,_, score, finished = env.make_move(action.item())
        reward = float(score[1])
        # print(reward)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()

        # if t > 15:
        #     done == True
        # else:
        #     done == False
        # print(done)
        # print(current_screen)
        # print(last_screen)
        # if not done:
        #     next_state = current_screen
        # else:
        #     next_state = None

        next_state = current_screen

        # Store the transition in memory
        # print(state)
        # print(state.size())
        # print(action)
        # print(action.size())
        # print(next_state)
        # # print(next_state.size())
        # print(reward)
        # print(reward.size())
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # print(state)
        # print(reward)
        # print(finished)
        # print(type(finished))


        # print(env.state()[0])

        # Perform one step of the optimization (on the target network)
        
        if finished:
            print(score)
            episode_durations.append(reward)
            plot_durations()
            # print(env.state()[0])
            # if num_episodes -  i_episode <= 100:
            comp_scores.append(score[0])
            user_scores.append(score[1])
            break

        optimize_model()
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())




print('Complete')

torch.save(policy_net.state_dict(), 'state_dict_final.pyt')


















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
            policynet = policynet / np.sum(policynet)
            # print(policynet)
            if math.isnan(policynet[0]):
                return torch.tensor([[random.choice(env.state()[2])]], device=device, dtype=torch.long)
            else:
                policymax = np.random.choice(64, p=policynet)
                # print(policymax)

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


new_evaluate_model(10)




# def s(iterations):



# plt.plot(horiz, vert)
# plt.show()





