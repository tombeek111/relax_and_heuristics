import numpy as np
from pulp import *
from itertools import product
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




prob += lpSum(x[u,v] * w[v] * d[u,v] for u,v in product(n,n)), "Minimize costs"

for u in n:
    prob += lpSum(x[u,v] for v in n) == 1
    
for u in n:
    for v in n:
        prob += x[u,v] <= y[v]
        
prob += lpSum(y[u] for u in n) <= len(p)

# Write problem to LP file
prob.writeLP("operatingRoomAssignment.lp")

# Solve problem using CPLex
solver = CPLEX()
prob.setSolver(solver)
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
#for v in prob.variables():
#    print(v.name, "=", v.varValue)
