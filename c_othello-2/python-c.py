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

global result

result = []

def output_reader(proc):
    # for line in iter(proc.stdout.readline, b''):
    #     # print('got line: {0}'.format(line.decode('utf-8')), end='')
    #     result.append( ('{0}'.format(line.decode('utf-8'))))
    # while True:
    line = proc.stdout.readline()
    result.append( ('{0}'.format(line.decode('utf-8'), end='')))
    # if not line: break
    

proc = subprocess.Popen(['./compiledothello.c'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE)







# t = threading.Thread(target=output_reader, args=(proc,))
# t.start()

# time.sleep(2)

count = 0
proc.stdin.write('3e'.encode())
# print(proc.stdin.read())
proc.stdin.close()
# proc.stdin.open()
# proc.stdin.write('4f'.encode())
# proc.stdin.close()
for i in range(0,30):

    output_reader(proc)
    count+=1
    if result != []:
        print(result)
        print(count)
        
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