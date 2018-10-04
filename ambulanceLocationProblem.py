import numpy as np
from pulp import *
#from numpy import *

# Open data set file
F = open('./Ambulance instances/Region14.txt','r') 

[n,p] =[range(0, int(s)) for s in F.readline().split()]

w = [int(s) for s in F.readline().split()]

d = np.empty([len(n),len(n)])

for i in n:
    d[i] = [int(s) for s in F.readline().split()]
    
# Initialize minimization problem
prob = LpProblem("Ambulance location problem",LpMinimize)

x = LpVariable.dict("x",(n,n),0,1,LpBinary)


y = LpVariable.dict("x",n,0,1,LpBinary)

