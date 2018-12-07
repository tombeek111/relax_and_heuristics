# -*- coding: utf-8 -*-
import numpy as np
import pulp
from itertools import product
from simulatedAnnealing import SAInstance
import copy 
class Solver:
    def __init__(self):
        self.verbose = True
        
        
    def do_branch_cut(self):
        global_ub = np.inf
        global_lb = np.inf
        best_solution = None
        
        P = SubProblem(self)
  
        
        
        active_nodes = [P]
        while active_nodes:
            P = active_nodes[0]
            del active_nodes[0]
            try:
                
                #Get lower bound, keep adding cuts until it is not possible
                first = True
                while True:
                    
                    
                    P.create_lp()
                    if not first:
                        cuts_added = self.add_cuts(P)
                        if not cuts_added:
                            break
                        
                        
                    P.compute_lb()
                    
                    if P.is_integer or P.lb >= global_ub:
                        break
                    
                        
                    
                    first = False
                
                #Get upper bound
                if P.is_integer:
                    #P is integer, so lb = ub
                    P.ub = P.lb
                    P.y = P.y_relax
                else:
                    P.compute_ub()
                
            except InfeasibleSubproblem:
                #pruned by infeasibility
                continue
            
            print('{0} {1}'.format(P.lb,P.ub))
            
            if P.lb >= global_ub:
                #Pruned because of too high lb
                continue
            
            if P.ub <= global_ub:
                global_ub = P.ub
                best_solution = P.y
                print('found new global ub')
                #Found best global ub
            
            
            if P.lb == P.ub:
                #Pruned by optimality
                print('lb = ub')
                continue
            
            
            P1,P2 = self.branch(P)
            active_nodes.extend([P1,P2])
            print('branch. Length {0}'.format(len(active_nodes)))
        self.solution = best_solution
        self.objective = global_ub

    
    def add_cuts(self,P):
        #Add cuts to P.P
        
        #If cuts are added, return True
        return False

    def branch(self,P):
        P1 = SubProblem(self)
        P1.y_one = copy.copy(P.y_one)
        P1.y_zero = copy.copy(P.y_zero)
        
        P2 = SubProblem(self)
        P2.y_one = copy.copy(P.y_one)
        P2.y_zero = copy.copy(P.y_zero)
        
        #branch on closest to one
        
        
        y_branch = max(P.y_noninteger, key=P.y_noninteger.get)
        
        #set fixed y
        P1.y_one.append(y_branch)
        P2.y_zero.append(y_branch)
        
        return P1,P2
       

class InfeasibleSubproblem(BaseException):
    pass

class Problem():
    def loadFile(self,filename):
        with open(filename,'r') as f:
            [self.n,self.p] =[int(s) for s in f.readline().split()]
            self.N = range(self.n)
            self.w = [int(s) for s in f.readline().split()]
            self.D = np.empty([self.n,self.n])
            for i in self.N:
                self.D[i] = [int(s) for s in f.readline().split()]

class SubProblem():
    def __init__(self,solver):
        self.is_integer = False
        self.solver = solver
        self.ub = None
        self.lb = None
        self.y_one = []
        self.y_zero = []
        
    def compute_lb(self):
        
        self.solve_relaxation()
        
    def solve_relaxation(self):
        self.P.solve()
        self.P.roundSolution()
        self.lb = pulp.value(self.P.objective)
        self.y_relax = {key:pulp.value(self.y_var[key]) for key in self.y_var}
        
        #Check if integer
        self.is_integer=True
        self.y_noninteger = {}
        for key,value in self.y_relax.items():
            if not value.is_integer():
                self.is_integer = False
                self.y_noninteger[key] = value
     
     

    
    def compute_ub(self):
        #Hier greedy solution of sim/ann. Schrijf waardes naar self.ub en self.y

        
        saInstance = SAInstance(problem.N,problem.w,problem.D,problem.p)
        solution = saInstance.greedy(self.y_one,self.y_zero)
        
        self.ub = solution.objective()
        self.y = solution.solution
        
    def create_lp(self):
        problem = self.solver.problem
        
        
        N = problem.N
        #Create LP object
        P = pulp.LpProblem("Ambulance location problem",pulp.LpMinimize)
        
        #Create variables
        x = pulp.LpVariable.dict("x",(N,N),0,1)
        y = pulp.LpVariable.dict("y",N,0,1)
        
        #set fixed vars
        for y_key in self.y_one:
            y[y_key] = float(1) #float so is_integer can be called later
            
        for y_key in self.y_zero:
            y[y_key] = float(0) #float so is_integer can be called later
            
            
        self.x_var = x
        self.y_var = y
        #set cost function
        P += pulp.lpSum(x[u,v] * problem.w[u] * problem.D[u,v] for u,v in product(N,N))
        
        #Add constraints
        #serve each city
        for u in N:
            P += pulp.lpSum(x[u,v] for v in N) == 1
            
        #Only select an edge(u,v) if there is an ambulance post at v
        for u,v in product(N,N):
            P += x[u,v] <= y[v]
        
        #Max p ambulance posts
        P += pulp.lpSum(y[u] for u in N) <= problem.p
        
        
        
        
        
       
        
        self.P = P
    
    
        
if __name__ == '__main__':
    problem = Problem()
    problem.loadFile('solutions/pmed2.txt')
    solver = Solver()
    solver.problem = problem
    
    solver.do_branch_cut()
    print('Solution {0}'.format(solver.objective))
    
