import subprocess
import sys
import time
import threading
import random
import numpy as np
import math


# To import:

# import sys
# sys.path.append(".")
# from cothelloago import crunner
# game = crunner.Move() #Othello game

class Move:

    def __init__ (self):
        self.board = "                           O@      @O                           \n"
        self.result = []
        # self.moves = ['3e', '4f', '5c', '6d']
        self.moves = [20,29,34,43]
        self.score = (0,0)
        self.over = 0
        self.actionNumber = 64
        self.boardArray = self.board_array()
    
    def reset(self):
        self.board = "                           O@      @O                           \n"
        self.result = []
        # self.moves = ['3e', '4f', '5c', '6d']
        self.moves = [20,29,34,43]
        self.score = (0,0)
        self.over = 0
        self.actionNumber = 64
        self.boardArray = self.board_array()

    def output_reader(self, proc):
        self.result = []
        while 1:
            line = proc.stdout.readline()
            if len(line) > 0: # If there is anything on the line

                self.result.append( ('{0}'.format(line.decode('utf-8'), end='')))
            else:
                return self.result

    def make_move(self, move):
        """Takes a move (number (0,63)) and returns the same as state"""

        proc = subprocess.Popen(["./compiledothello.c"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        move = self.encode_move(move)
        input_write = self.board + move
        proc.stdin.write(input_write.encode())
        proc.stdin.close()
        self.output = self.output_reader(proc)
        # print(self.output)
        self.board = self.output[0]
        self.moves = []
        # for item in self.output:
        #     self.moves.append(item)
        self.moves = self.output[1:-2]
        self.moves = [s.strip('\n') for s in self.moves]
        self.moves = [self.decode_move(a) for a in self.moves]
        self.score = eval(self.output[-2].strip('\n'))
        self.over = int(self.output[-1])
        self.boardArray = self.board_array()
        return (self.boardArray, self.board, self.moves, self.score, self.over)

    def board_array(self):
        boardArray = np.empty([8,8])
        count = 0
        numboard = [None] * 64
        listboard = list(self.board)
        for i in range(64):
            if listboard[i] == ' ':
                numboard[i] = 0
            if listboard[i] == '@':
                numboard[i] =-1
            if listboard[i] == 'O':
                numboard[i] = 1
        for i in range(0,8):
            for j in range(0,8):
                boardArray[i,j] = numboard[count]
                count += 1
        return boardArray


    def state(self):
        """Returns a numpy array of the board, the board string, the available moves in a number (0-63), 
        the score (opponent, self), and whether the game is over (0,1)"""
        return [self.boardArray, self.board, self.moves, self.score, self.over]


    def encode_move(self, move):
        ''' Encode a move into the format that the c program wants for example move 0 is 1a'''
        move = move+1
        row = math.ceil(move/8)
        column = move - 8*(row -1)

        start = ord('a') - 1
        letter = chr(start + column)

        return str(row) + str(letter)


    def decode_move(self, move):
        ''' Decode a move into an array position for example move 1a is 0'''
        row = int(move[0]) -1
        column = ord(move[1]) - ord('a')
        decoded = 8*row + column
        return decoded

    #methods for alpha zero
    
    def getInitBoard(self):
        return self.boardArray

    def getBoardSize(self):
        return (8,8)
    
    def getActionSize(self):
        return 65

    def getNextState(self, action):
        
        board = self.make_move(action)[0]
        return (board, 1)
    
    def getValidMoves(self, board, player):
        valids = [0]*self.getActionSize()
        lett_num = {"a": 1,
                    "b": 2,
                    "c": 3,
                    "d": 4,
                    "e": 5,
                    "f": 6,
                    "g": 7,
                    "h": 8
                    }
        if len(self.moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for move in self.moves:
            valids[8*int[move[0] + int[lett_num[move[1]]]]] = 1
        return np.array(valids)
    
    def getGameEnded(self, board, player):
        if self.over:
            if self.score[0]>self.score[1]:
                return 1
            else:
                return -1
        return self.over

    def getCanonicalForm(self, board, player):


if __name__ == "__main__":

    move = Move()


    # print(move.decode_move('3e'))
    # print(move.decode_move('4f'))
    # print(move.decode_move('5c'))
    # print(move.decode_move('6d'))
    print(move.state()[2])

    print(move.board_array())
    
    moves = [20,29,34,43]
    score = (0,0)
    over = '0'

    while over == '0' and moves != []:
        _, board, moves, score, over = move.make_move(moves[random.randint(0,len(moves)-1)])
        # if over == '1':
        print(board, moves, score, over)
    
    print(move.board_array())
        