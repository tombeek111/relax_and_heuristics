import numpy as np
import pulp
from itertools import product
import sys

##% Load file
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print('Using command line file: {0}'.format(filename))
else:
    filename = './Ambulance instances/Region05.txt'
    print('Using hardcoded filename: {0}'.format(filename))
    
with open(filename,'r') as f:
    [n,p] =[int(s) for s in f.readline().split()]
    N = range(n)
    w = [int(s) for s in f.readline().split()]
    D = np.empty([n,n])
    for i in N:
        D[i] = [int(s) for s in f.readline().split()]

#%% Create LP problems
def createLp(N,p,w,D,binary=True):
    #Create LP object
    P = pulp.LpProblem("Ambulance location problem",pulp.LpMinimize)
    
    #Create variables
    if binary:
        x = pulp.LpVariable.dict("x",(N,N),0,1,pulp.LpBinary)
        y = pulp.LpVariable.dict("y",N,0,1,pulp.LpBinary)
    else:
        x = pulp.LpVariable.dict("x",(N,N),0,1)
        y = pulp.LpVariable.dict("y",N,0,1)
        
    #set cost function
    P += pulp.lpSum(x[u,v] * w[v] * D[u,v] for u,v in product(N,N))
    
    #Add constraints
    #serve each city
    for u in N:
        P += pulp.lpSum(x[u,v] for v in N) == 1
        
    #Only select an edge(u,v) if there is an ambulance post at v
    for u,v in product(N,N):
        P += x[u,v] <= y[v]
    
    #Max p ambulance posts
    P += pulp.lpSum(y[u] for u in N) <= p
    
    return P

problems = {'binary' : createLp(N,p,w,D,True),'relaxed' : createLp(N,p,w,D,False)}
values = {}
for name,P in problems.items():
    P.setSolver(pulp.CPLEX())
    status = P.solve()
    
    print('Solved {0} problem. Status: {1}'.format(name,status))
    if name == 'binary':
        #Round solutions to change values like 0.999999 to 1
        P.roundSolution()
    
    value = pulp.value(P.objective)
    print('{0} problem objective value: {1}'.format(name,value))
    values[name] = value
    
print('Ratio Relaxed/Binary: {0:.3f}'.format(values['relaxed']/values['binary']))
