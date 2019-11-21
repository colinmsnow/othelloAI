import subprocess
import sys
import time
import threading
import random
import numpy as np

class Move:

    def __init__ (self):
        self.board = "                           O@      @O                           \n"
        self.result = []
        self.moves = ['3e', '4f', '5c', '6d']
        self.score = (0,0)
        self.over = '0'
        self.actionNumber = 64
    
    def reset(self):
        self.board = "                           O@      @O                           \n"
        self.result = []
        self.moves = ['3e', '4f', '5c', '6d']
        self.score = (0,0)
        self.over = '0'
        self.actionNumber = 64

    def output_reader(self, proc):
        self.result = []
        while 1:
            line = proc.stdout.readline()
            if len(line) > 0: # If there is anything on the line

                self.result.append( ('{0}'.format(line.decode('utf-8'), end='')))
            else:
                return self.result

    def make_move(self, move):

        proc = subprocess.Popen(['./compiledothello.c'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
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
        self.score = self.output[-2].strip('\n')
        self.over = self.output[-1]
        return (self.board, self.moves, self.score, self.over)

    def board_array(self):
        boardArray = np.empty([8,8])
        count = 0
        numboard = [None] * 64
        listboard = list(self.board)
        for i in range(64):
            # print(self.board[i])
            # print(type(self.board[i]))
            # numboard[i] = listboard[i]
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
        return [self.board, self.moves, self.score, self.over]



if __name__ == "__main__":

    move = Move()

    print(move.board_array())
    
    moves = ['3e', '4f', '5c', '6d']
    score = (0,0)
    over = '0'

    while over == '0' and moves != []:
        board, moves, score, over = move.make_move(moves[random.randint(0,len(moves)-1)])
        # if over == '1':
        print(board, moves, score, over)
    
    print(move.board_array())
        