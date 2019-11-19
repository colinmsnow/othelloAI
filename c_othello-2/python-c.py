import subprocess
import sys
import time
import threading
import random

class Move:

    def __init__ (self):
        self.board = "                           O@      @O                           \n"
        self.result = []

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

move = Move()

board, moves, score, over = move.make_move('3e')

while over == '0' and moves != []:
    board, moves, score, over = move.make_move(moves[random.randint(0,len(moves)-1)])
    # if over == '1':
    print(board, moves, score, over)
        