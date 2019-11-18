#!/usr/bin/python -u
# import subprocess

# input_data = '3e'
# p = subprocess.Popen(['./compiledothello.c'], 
#                      stdin=subprocess.PIPE,stdout=subprocess.PIPE)
# # p.stdin.write(str.encode(input_data))
# # print(p.stdout.readline())

# p.stdin.write(str.encode(input_data))
# p.stdin.close()
# result = p.stdout.read()
# print (result)
# p.wait()

import subprocess
import sys
import time
import threading
import random

global result

class Move:

    def __init__ (self):
        self.board = "                           O@      @O                           \n"
        self.result = []

    def output_reader(self, proc):
        # for line in iter(proc.stdout.readline, b''):
        #     # print('got line: {0}'.format(line.decode('utf-8')), end='')
        #     result.append( ('{0}'.format(line.decode('utf-8'))))
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
print(board, moves, score, over)
# board, moves, score = move.make_move('4f')
# print(board, moves, score)
# print(type(over))
while over == '0' and moves != []:
    board, moves, score, over = move.make_move(moves[random.randint(0,len(moves)-1)])
    print(board, moves, score, over)
        
# board = Board()

# proc = subprocess.Popen(['./compiledothello.c'],
#     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# move = '3e'
# input_write = board.board + move
# proc.stdin.write(input_write.encode())
# proc.stdin.close()
# output_reader(proc)
# for i in range(0,30):

#     output_reader(proc)
#     count+=1
#     if result != []:
#         print(result)
#         print(count)








# # t = threading.Thread(target=output_reader, args=(proc,))
# # t.start()

# # time.sleep(2)

# count = 0

# proc.stdin.write('3e'.encode())
# # print(proc.stdin.read())
# proc.stdin.close()
# # proc.stdin.open()
# # proc.stdin.write('4f'.encode())
# # proc.stdin.close()
# for i in range(0,30):

#     output_reader(proc)
#     count+=1
#     if result != []:
#         print(result)
#         print(count)
        
    # result = []
    # input("press enter")
# for i in range(0,1000):
#     if result != []:
#         print(result)
# time.sleep(2)
# while 1:
#     proc.stdin.write('4f'.encode())
#     if result != []:
#         print(result)
#         break
    # result = []
    # input("press enter")
    # time.sleep(.1)
# t.join()

# time.sleep(1)
# proc.stdin.write('3e\n\n'.encode())
# proc.stdout.read()
# # proc.stdin.write(str.encode('3e\n'))
# # proc.stdin.write(str.encode('3e\n'))
# # proc.stdin.write("mark\n")
# # proc.stdin.write("luke\n")
# # proc.stdin.close()


# print (proc.communicate('3e\n\n'.encode()))
# # for line in iter(proc.stdout.readline,''):
#    print (line.rstrip())


# while proc.returncode is None:
#     proc.poll()

# print ("I got back from the program this:\n{0}".format(proc.stdout.read()))