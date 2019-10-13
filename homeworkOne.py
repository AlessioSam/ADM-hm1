#Introduction

#0 Solve Me First

def solveMeFirst(a,b):
	# Hint: Type return a+b below
    return a+b

num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)

#-----------------------------------------

#1 Say "Hello, World!" With Python

print("Hello, World!")

#---------------------------------------------------

#2 Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())

    if n%2!=0 or n==0 :
        print("Weird")
    else: 
        if 2<=n<=5:
            print("Not Weird")
        elif 6<=n<=20:
            print('Weird')
        else:
            print('Not Weird') 			

#--------------------------------------

#3 Arithmetic Operators

if __name__ == '__main__':
   a = int(input())
   b = int(input())

print(a+b)
print(a-b)
print(a*b)
#-------------------------------------------

#4 Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a//b)
print(a/b)

#-----------------------------------------

#5 Loops

if __name__ == '__main__':
    n = int(input())

for i in range(n):
    print(i**2)
    
#----------------------------------------------

#6 Write a function

def is_leap(year):
    leap = False
    
    if year%4==0:
        leap=True
        if year%100==0: 
            leap = False
            if year%400==0:
                leap=True    
    
    return leap

#-----------------------------------------------
    
#7 Print Function

if __name__ == '__main__':
    n = int(input())

l=str('')
for i in range(n):
   l=l+str(i+1)

print(l)    

####################################################
####################################################

# DATA TYPES CHALLENGE

#1 List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

m=[]
c=0
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if (i+k+j!=n):
                m.append([])
                m[c]=[i,j,k]
                c=c+1
                               
print(m)

#---------------------------------------------------

#2 Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())


arr=sorted(arr)
ok=int()
for i in range(len(arr)-1,-1,-1):
    if (arr[-1]>arr[i]):
        ok=arr[i]
        break

print(ok)

#------------------------------------------------------

#3 Nested Lists

#i looked part of solutions

sc=[]
nm=[]
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        sc.append(score)
        nm.append(name)

scoreord=sorted(sc)
low=0
for i in range(0, len(scoreord)):
    if(scoreord[i]>scoreord[0]):
        low=scoreord[i]
        break


namelow=[]
for i in range(0,len(sc)):
    if(sc[i]==low):
        namelow.append(nm[i])
namelow2=sorted(namelow)

for i in range(0, len(namelow2)):
    print(namelow2[i])

#-----------------------------------------------------
    
#4   Finding the percentage

if __name__ == '__main__':


    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()


add=0
for i in (student_marks[query_name]):
    add+=i

print('%.2f' %(add/3))

#--------------------------------------------------------

#5 Lists

#Looking solutions, the request is hard to understand

if __name__ == '__main__':
    N = int(input())
    listt = []
    while N > 0:
        q = list(map(str,input().split()))
        if q[0] == "print":
            print(listt)
        elif q[0] == "insert":
            listt.insert(int(q[1]),int(q[2]))
        elif q[0] == "remove":
            listt.remove(int(q[1]))
        elif q[0] == "append":
            listt.append(int(q[1]))
        elif q[0] == "sort":
            listt.sort()
        elif q[0] == "pop":
            listt.pop()
        elif q[0] == "reverse":
            listt.reverse()
        N -= 1

#-----------------------------------------------------------
        
#6  Tuples
        
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

    print(hash(tuple(integer_list)))

##############################################################
#################################################################

#  STRINGS CHALLENGE

#1   sWAP cASE

def swap_case(s):
    s=s.swapcase()
   

    return s

#---------------------------------------------------------------
    
#2 String Split and Join


def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line

#---------------------------------------------------------------
    
#3 What's Your Name?

def print_full_name(a, b):
    print("Hello",a, str(b)+"! You just delved into python.")
    
 #--------------------------------------------------------------

#4 Mutations   
 
def mutate_string(string, position, character):
    string=string[:position]+character+string[position+1:]
    return string
    
#-------------------------------------------------------------

#5 Find a string

def count_substring(string, sub_string):
    c=0
    for i in range(len(string)-len(sub_string)+ 1):
        if string[i:i+len(sub_string)]==sub_string:
            c+=1
    

    return c

#----------------------------------------------------------------
 
#6 String Validators

#I LOOKED AT THE SOLUTION FOR AN ALTERNATIVE WAY OF WRITING THE CYCLE
if __name__ == '__main__':
    s = input()

print(any([char.isalnum() for char in s]))
print(any([char.isalpha() for char in s]))
print(any([char.isdigit() for char in s]))
print(any([char.islower() for char in s]))
print(any([char.isupper() for char in s]))    
    
#---------------------------------------------------------------

#7 Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

# Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

# Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

# Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

# Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
    
#--------------------------------------------------------

#8  Text Wrap

def wrap(string, max_width):
    wrap=textwrap.fill(string, max_width)
    return wrap

#--------------------------------------------------------
    
#9 Designer Door Mat

#Looking solution
alt, lun = map(int, input().split())
for i in range(0, alt//2):
    s = '.|.'*(i*2 + 1)
    print(s.center(lun,'-'))
print('WELCOME'.center(lun, '-'))
for i in range(alt//2-1,-1,-1):
    s = '.|.'*(i*2+1)
    print(s.center(lun,'-'))
    
#-----------------------------------------------------------------

#10  String Formatting
    
def print_formatted(number):
    form = len("{0:b}".format(number))
   
    for i in range(1, number + 1):
        print("{0:{form}d} {0:{form}o} {0:{form}X} {0:{form}b}".format(i, form=form))


#------------------------------------------------------------------
        
#11 Capitalize!

#I looked part of solution to see how to enter the spaces
def solve(s):
    s=' '.join(s.capitalize() for s in s.split(' '))
    return s        

 
#------------------------------------------------------------------

#12 Alphabet Rangoli

#I looked Solution
import string
def print_rangoli(size):
    a = string.ascii_lowercase
    r = []
    for i in range(size):
        s = "-".join(a[i:size])
        r.append((s[::-1]+s[1:]).center(4*size-3, "-"))
    print('\n'.join(r[:0:-1]+r))    
     
#-------------------------------------------------------------------
    
#13  The Minion Game

def minion_game(string):
    length = len(string)

    the_vowel = "AEIOU"

    kiven = 0
    stuart = 0
    for i in range(length):
        if string[i] in the_vowel:
            kiven = kiven + length - i
        else:
            stuart = stuart + length - i

    if (kiven > stuart):
        print ("Kevin", kiven)
    elif (kiven < stuart):
        print ("Stuart", stuart)
    else:
        print ("Draw")   
        
#-------------------------------------------------------------

#14 Merge the Tools!

def merge_the_tools(string, k):
    num = int(len(string) / k)

    for i in range(num):
        x = string[i*k : (i+1)*k]
        eq = ""

        for j in x:
            if (j not in eq):
                eq+=j

       
        print(eq) 
        
#----------------------------------------------------------------
#################################################################
###############################################################

# SETS CHALLENGE      

#1  Introduction to Sets

def average(array):
   a= set(array)
   m=sum(a)/len(a)
   return m

#-----------------------------------------------------------------

#2  No Idea!
   

(n,m) = map(int, input().split())
arr = map(int, input().split())

A = set(map(int, input().split()))
B = set(map(int, input().split()))


happ=0

for i in arr:
    if i in A:
        happ+=1
    elif i in B:
        happ-=1

print (happ)  

#-------------------------------------------------------------------
#now I use the numbering found on slack, sorry

# Exercise 30 - Sets - Symmetric Difference
#Solution. I don't understand request...
A,B= [set(input().split()) for _ in range(4)][1::2]


print('\n'.join(sorted(A.symmetric_difference(B), key=int)))

# Exercise 31 - Sets - Set .add()
N=int(input())
country=set()
for _ in range(1,N):
    country.add(input())

print(len(country))

# Exercise 32 - Sets - Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))

for i in range(int(input())):
    eval('s.{0}({1})'.format(*input().split()+['']))

print(sum(s))



# Exercise 33 - Sets - Set .union() Operation
a=input()
A=set(map(int,input().split()))
b=input()
B=set(map(int,input().split()))



print (len(A.union(B)))

# Exercise 34 - Sets - Set .intersection() Operation
a=input()
A=set(map(int,input().split()))
b=input()
B=set(map(int,input().split()))



print (len(A.intersection(B)))

# Exercise 35 - Sets - Set .difference() Operation
a=input()
A=set(map(int,input().split()))
b=input()
B=set(map(int,input().split()))



print (len(A.difference(B)))

# Exercise 36 - Sets - Set .symmetric_difference() Operation
a=input()
A=set(map(int,input().split()))
b=input()
B=set(map(int,input().split()))



print (len(A.symmetric_difference(B)))

# Exercise 37 - Sets - Set Mutations
length=int(input())
ins=set(map(int, input().split()))
N=int(input())

for i in range(N):
    (p, q)=input().split()
    ins2=set(map(int,input().split()))
    if p=='intersection_update':
        ins.intersection_update(ins2)
    elif p=='update':
        ins.update(ins2)
    elif p=='symmetric_difference_update':
        ins.symmetric_difference_update(ins2)
    elif p=='difference_update':
        ins.difference_update(ins2)
print (sum(ins))

# Exercise 38 - Sets - The Captain's Room
K=input()
set1=set()
set2=set()

for i in (input().split()):
    if i not in set1:
        set1.add(i)
    else:
        set2.add(i)
set1.difference_update(set2)
print(set1.pop())

# Exercise 39 - Sets - Check Subset
for _ in range(int(input())):
    a=input()
    A=set(input().split())
    b=input()
    B=set(input().split())
    print(A.issubset(B))

# Exercise 40 - Sets - Check Strict Superset
supers=set(input().split())

B = set()

for _ in range(int(input())):
    B |= set(input().split()) #i've look on internet for |=

print (not bool(B ^ supers))

# Exercise 41 - Collections - collections.Counter()
from collections import Counter
shoes=int(input())
sizes=Counter(map(int, input().split()))
cust=int(input())

sale=0
for _ in range(cust):
    s,p=map(int,input().split())
    if sizes[s]:
        sale+=p
        sizes[s]=sizes[s]-1


print(sale)

# Exercise 42 - Collections - DefaultDict Tutorial
from collections import defaultdict
n,m=map(int,input().split())


d=defaultdict(list)
for i in range(0,n):
    d[input()].append(i+1)

b=[]
for j in range(0,m):
    b=b+[input()]

for i in b:
    if(i in d):
        print(" ".join(map(str,d[i])))    
    else:
        print(-1)

# Exercise 43 - Collections - Collections.namedtuple()
#looking solution to understand 'namedtuple'
from collections import namedtuple

n=int(input())
col=input().split()

markstot=0
for _ in range(n):
    students = namedtuple('student', col)
    MARKS, CLASS, NAME, ID = input().split()
    student = students(MARKS, CLASS, NAME, ID)
    markstot += int(student.MARKS)
print('{:.2f}'.format(markstot / n))

# Exercise 44 - Collections - Collections.OrderedDict()
from collections import OrderedDict

dic=OrderedDict()
n=int(input())

for _ in range(n):
    item=input().split()
    iprice=int(item[-1])
    iname=" ".join(item[:-1])

    if dic.get(iname):
        dic[iname]+=iprice
    else:
        dic[iname]=iprice    

for j in dic.keys():
    print(j, dic[j])


# Exercise 45 - Collections - Word Order
from collections import * 

n=int(input())
dic=OrderedDict()

for _ in range(n):
    read=input()
    if read in dic:
        dic[read]+=1
    else:
        dic[read]=1

print(len(dic))
print (' '.join(map(str, dic.values())))

# Exercise 46 - Collections - Collections.deque()
from collections import deque

f=deque()

for _ in range(int(input())):
    com = input().strip().split()
    if (com[0] == 'append'):
        f.append(com[1])
    elif (com[0] == 'pop'):
        f.pop()
    elif (com[0] == 'popleft'):
        f.popleft()
    elif (com[0] == 'appendleft'):
        f.appendleft(com[1])

print (' '.join(f))


# Exercise 47 - Collections - Company Logo
import math
import os
import random
import re
import sys

import collections

if __name__ == '__main__':
    s = input().strip()

s=sorted(s)
scont=collections.Counter(s).most_common()
scont=sorted(scont, key=lambda x: (x[1] * -1, x[0])) #solution for key=lambda.....
for i in range(0, 3):
    print(scont[i][0], scont[i][1])

# Exercise 48 - Collections - Piling Up!
#NO 

# Exercise 49 - Date time - Calendar Module
import calendar
mese,gg,anno=map(int,input().strip().split())

dayweek=calendar.day_name[calendar.weekday(anno,mese,gg)].upper()
print(dayweek)

# Exercise 50 - Date time - Time Delta
import math
import os
import random
import re
import sys

from datetime import datetime
# Complete the time_delta function below.
def time_delta(t1, t2):
        uno = datetime.strptime(t1, "%a %d %b %Y %X %z")
        due = datetime.strptime(t2, "%a %d %b %Y %X %z")
        return str(int(abs((uno-due).total_seconds()))) 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# Exercise 51 - Exceptions -
    
n=int(input())

for _ in range(n):
    a,b=input().split()
    try:
        print(int(a)//int(b))
    except Exception as e:
        print ("Error Code: " + str(e)) #solution for str()

# Exercise 52 - Built-ins - Zipped!
n,x=map(int,input().split())


sub=list()

for i in range(x): 
    sub.append(map(float, input().split())) #matrix

stud=zip(*sub)

for i in stud:
    print(sum(i)/x)
        
# Exercise 53 - Built-ins - Athlete Sort
if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

arr = sorted(arr, key=lambda col: col[k])
for i in range(n):  
    print(" ".join(str(x) for x in arr[i]))   
    
# Exercise 54 - Built-ins - Ginorts
#I looked solution, I had no idea how to do it fast but i was sure there was a 
#good way better than mine.
print(*sorted(input(), key=lambda c: (-ord(c) >> 5, c in '02468', c)), sep='')
    
# Exercise 55 - Map and lambda function
cube = lambda x: x ** 3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    seq= [0, 1]
    for i in range(2, n):
        seq.append(seq[i-1] + seq[i-2])
        
    return(seq[0:n])  #i addedd [0:n] after looking at the solution
cube = lambda x: x ** 3



if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
    
#I NEVER STUDIED REGEX, XML AND DECORATORS... I'LL STUDY AND I'LL 
#COMPLETE THEM IN NEXT WEEKS   
    
# Exercise 56 - Regex - Detect Floating Point Number
# Exercise 57 - Regex - Re.split()
# Exercise 58 - Regex - Group(), Groups() & Groupdict()
# Exercise 59 - Regex - Re.findall() & Re.finditer()
# Exercise 60 - Regex - Re.start() & Re.end()
# Exercise 61 - Regex - Regex Substitution
# Exercise 62 - Regex - Validating Roman Numerals
# Exercise 63 - Regex - Validating phone numbers
# Exercise 64 - Regex - Validating and Parsing Email Addresses
# Exercise 65 - Regex - Hex Color Code
# Exercise 66 - Regex - HTML Parser - Part 1
# Exercise 67 - Regex - HTML Parser - Part 2
# Exercise 68 - Regex - Detect HTML Tags, Attributes and Attribute Values
# Exercise 69 - Regex - Validating UID
# Exercise 70 - Regex - Validating Credit Card Numbers
# Exercise 71 - Regex - Validating Postal Codes
# Exercise 72 - Regex - Matrix Script
# Exercise 73 - Xml - XML 1 - Find the Score
# Exercise 74 - Xml - XML 2 - Find the Maximum Depth    
# Exercise 75 - Closures and decorators - Standardize Mobile Number Using Decorators
# Exercise 76 - Closures and decorators - Decorators 2 - Name Directory


# Exercise 77 - Numpy - Arrays
def arrays(arr):
    a=list(reversed(arr))
    a1=numpy.array(a,float)
    
    return a1

# Exercise 78 - Numpy - Shape and Reshape
import numpy

arr=input().split()


arr = numpy.array(arr, int)

print (numpy.reshape(arr,(3,3)))

# Exercise 79 - Numpy - Transpose and Flatten
import numpy

n,m=map(int,input().split())

arr = list()
for i in range(n):
    arr.append(input().split())

matrice=numpy.reshape(numpy.array(arr, int), (n,m))

print(matrice.transpose())
print(matrice.flatten())

# Exercise 80 - Numpy - Concatenate
import numpy

n,m,p=map(int,input().split())


narr=[list(map(int, input().split())) for i in range(n)]
marr=[list(map(int, input().split())) for i in range(m)]

arr1=(numpy.array(narr))
arr2=(numpy.array(marr))

print(numpy.concatenate((arr1,arr2),axis=0))

# Exercise 81 - Numpy - Zeros and Ones
import numpy

mat=tuple(map(int,input().strip().split()))

print(numpy.zeros(mat, dtype = numpy.int))
print(numpy.ones(mat, dtype=numpy.int))

# Exercise 82 - Numpy - Eye and Identity
import numpy

r,c=map(int,input().split())
numpy.set_printoptions(sign=' ') #looking solution for the space, the output was rigth!
print(numpy.eye(r,c))

# Exercise 83 - Numpy - Array Mathematics
import numpy

n,m=map(int,input().split())

arr1= numpy.array([list(map(int,input().split())) for i in range(n)])
arr2= numpy.array([list(map(int,input().split())) for i in range(n)])


print (numpy.add(arr1,arr2))
print (numpy.subtract(arr1,arr2)    ) 
print (numpy.multiply(arr1,arr2) )   
print (arr1//arr2 )
print (numpy.mod(arr1,arr2) ) 
print (numpy.power(arr1,arr2)) 

# Exercise 84 - Numpy - Floor, Ceil and Rint
import numpy

a=numpy.array(list(map(float,input().split())))
numpy.set_printoptions(sign=' ')

print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))



# Exercise 85 - Numpy - Sum and Prod
import numpy

n,m=map(int,input().split())

a=numpy.array([input().split() for _ in range(n)],int)
print(numpy.prod(numpy.sum(a, axis = 0),axis=0))

# Exercise 86 - Numpy - Min and Max
import numpy

n,m = map(int, input().split())
a = numpy.array([input().split() for _ in range(n)],int)
print(numpy.max(numpy.min(a, axis=1), axis=0)) #solution, i had a problem with axis=1

# Exercise 87 - Numpy - Mean, Var, and Std
import numpy

n,m=map(int,input().split())


a = numpy.array([input().split() for _ in range(n)],int)

numpy.set_printoptions(sign=' ')
print(numpy.mean(a, axis=1))
print(numpy.var(a,axis=0))
print(numpy.around(numpy.std(a),12))

# Exercise 88 - Numpy - Dot and Cross
import numpy

n=int(input())

a= numpy.array([input().split() for _ in range(n)],int)
b= numpy.array([input().split() for _ in range(n)],int)

print(numpy.dot(a,b))

# Exercise 89 - Numpy - Inner and Outer
import numpy


a= numpy.array(list(map(int,input().split())))
b= numpy.array(list(map(int,input().split())))

print(numpy.inner(a,b))
print(numpy.outer(a,b))

# Exercise 90 - Numpy - Polynomials
import numpy

a = numpy.array(list(map(float, input().split())))
pos=float(input())
print( numpy.polyval(a, pos))

# Exercise 91 - Numpy - Linear Algebra
​import numpy

n=int(input())
mat= numpy.array( [input().split() for _ in range(n)],float)
numpy.set_printoptions(legacy='1.13') #solution for this line
print(round(numpy.linalg.det(mat),2))


# ===== PROBLEM2 =====
​
# Exercise 92 - Challenges - Birthday Cake Candles

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    off=0
    top=max(ar)
    for i in range(len(ar)):
       if (ar[i]==top):
            off+=1
    return off

   
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 93 - Challenges - Kangaroo
import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    
    for n in range(10000):
        if((x1+v1)==(x2+v2)):
            return "YES"
        x1+=v1
        x2+=v2
    return "NO"
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Exercise 94 - Challenges - Viral Advertising
import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    cum=[]
    cum.append(math.floor(5/2))
    for i in range(1,n,1):
        cum.append(math.floor(cum[i-1]/2)+cum[i-1])
    return sum(cum)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 95 - Challenges - Recursive Digit Sum
import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    N=0
    for i in range(0,len(n)):
        N+=int(n[i])
    f=str(N)*k
    
    
    
   
    n=0
    for i in range(0,len(f)):
        n+=int(f[i])
    
    if(n<10):
       
        return n
    else: 
           
        return superDigit(str(n), 1)
    
   
    


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 96 - Challenges - Insertion Sort - Part 1
import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    k=arr[-1]
    
    for i in range(n-2,-1,-1):
        if (arr[i]>k):
            arr[i+1]=arr[i]
            print (" ".join(str(f) for f in arr)) #looking solution for this line
        else:
            arr[i+1]=k
            print (" ".join(str(f) for f in arr))
            return
    
    arr[0] = k
    print (" ".join(str(f) for f in arr))
    return


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Exercise 97 - Challenges - Insertion Sort - Part 2 
 import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):

    for i in range(1, len(arr)):
        k=arr[i]
        j=i
        while j>0 and k<arr[j-1]:
            arr[j]=arr[j-1]
            j-=1
        arr[j]=k
        print (' '.join(str(j) for j in arr))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr) 
    