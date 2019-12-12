'''
Code from https://github.com/suragnair/alpha-zero-general
'''

import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# from cothello import crunnerago
# game = crunnerago.Move()   #TODO: move to different doc

'''
This module contains the OthelloNNet class, which is the convolutional neural net we will be using for our RL.
The CNN has:
 - 4 convolutional layers
 - 4 fully connected layers

 pytorch's dropout module randomly zeros values which prevents co-adaptation.
 By using batch normalization, no activation will get too high or too low.
 The network also uses ReLu as its activation function of choice.

 Note how pi, or the policy, is derived by an output size of the number of valid moves. softmax 
 will convert the output values to probabilities that sum to one.
 On the other hand, v, is a single value that is likely predicting the state of the game. the tanh function adjusts v.Ss

'''

class OthelloNNet(nn.Module):
    #TODO: make sure game has .getBoardSize() and .getActionSize()
    def __init__(self, game, args):
        super(OthelloNNet, self).__init__()
        # game params
        # self.board_x, self.board_y = game.getBoardSize()
        self.board_x, self.board_y = 8,8
        self.args = args
        # self.action_size = game.getActionSize()
        self.action_size = len(game.moves)

        #TODO: Look into args and figure out .num_channels()
        #Args 

        #TODO: Create flow diagram of neural network
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)