# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


def is_x(x):
    n = 0
    i = 0    
    while n < x:
        i += 1
        n = n + i
    if n == x:
        print ('need zhe me duo ci:'+str(i))
        for j in range(i):
            print ('+' + str(j+1),end='')
        print ('=' + str(x))
    elif (n - x) % 2 ==0:
        print ('need zhe me duo ci:'+str(i))
        for j in range(i):
            if (n-x) / 2 == j+1:
                print ('-' ,end ='')
            else:
                print ('+',end ='')
            print (str(j+1),end ='')
        print ('=' + str(x))
            
    elif  (i + 1) % 2 == 0:
        print ('need zhe me duo ci:'+str(i+2))
        for j in range(i+2):
            if (n-x - 1) / 2 == j+1:
                print ('-',end ='')
            elif j+1 == i+1:
                print ('+',end ='')
            elif j+1 == i+2:
                print ('-',end ='')
            else:
                print ('+',end ='')
            print (str(j+1),end ='')
        print ('=' + str(x))
        
    elif (i + 1) % 2 != 0:
        print ('need zhe me duo ci:'+str(i+1))
        for j in range(i+1):
            if (n-x + i+1 ) / 2 == j+1:
                print ('-',end ='')
            else:
                print ('+',end ='')
            print (str(j+1),end ='')
        print ('=' + str(x))
    


x = 7
x_in = input('please enter your x: ')
print ('your x is :',x_in)
x = int(x_in)
while x_in != 'q':
    is_x(x)
    x_in = input('please enter your x: ')
    print ('your x is :',x_in)
    x = int(x_in)
