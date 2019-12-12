import subprocess
import sys
import time
import threading
import random
import numpy as np
import math

class Move:
    ''' The game class which handles game mechanics and turns '''

    def __init__ (self):
        ''' Creates a move class and gives the initial state of the board (4 pieces in the middle)'''
        self.board = "                           O@      @O                           \n"
        self.result = []
        self.moves = [20,29,34,43]
        self.score = (0,0)
        self.over = '0'
        self.actionNumber = 64
        self.boardArray = self.board_array()
    
    def reset(self):
        ''' Resets the state for a new game'''
        self.board = "                           O@      @O                           \n"
        self.result = []
        self.moves = [20,29,34,43]
        self.score = (0,0)
        self.over = '0'
        self.actionNumber = 64
        self.boardArray = self.board_array()

    def output_reader(self, proc):
        ''' Read the output of the c program to a pythin list'''
        self.result = []
        while 1:
            line = proc.stdout.readline()
            if len(line) > 0: # If there is anything on the line
                self.result.append( ('{0}'.format(line.decode('utf-8'), end='')))
            else:
                return self.result

    def make_move(self, move):
        ''' Make the specified move and then run the c program to make an opposing move and update the board '''
        proc = subprocess.Popen(["./compiledothello.c"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        move = self.encode_move(move)
        input_write = self.board + move
        proc.stdin.write(input_write.encode())
        proc.stdin.close()
        self.output = self.output_reader(proc)
        self.board = self.output[0]
        self.moves = []
        self.moves = self.output[1:-2]
        self.moves = [s.strip('\n') for s in self.moves]
        self.moves = [self.decode_move(a) for a in self.moves]
        self.score = eval(self.output[-2].strip('\n'))
        self.over = int(self.output[-1])
        self.boardArray = self.board_array()
        return (self.boardArray, self.board, self.moves, self.score, self.over)

    def board_array(self):
        ''' Convert a board string to a numpy array '''
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
        """Returns a numpy array of the board, the board string, the available moves in a numbered list (0-63), 
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

if __name__ == "__main__":

    move = Move()
    print(move.state()[2])
    print(move.board_array())
    
    moves = [20,29,34,43]
    score = (0,0)
    over = '0'

    while over == '0' and moves != []:
        _, board, moves, score, over = move.make_move(moves[random.randint(0,len(moves)-1)])
        print(board, moves, score, over)
    
    print(move.board_array())
        