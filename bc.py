# -*- coding: utf-8 -*-
import numpy as np
import pulp
from itertools import product

class Solver:
    def do_branch_cut(self):
        global_ub = np.inf
        global_lb = np.inf
        
        
        P = SubProblem(self)
  
        
        
        active_nodes = [P]
        while active_nodes:
            P = active_nodes[0]
            
            try:
                
                while True:
                    P.compute_lb()
                    if P.is_integer or P.lb >= global_ub:
                        break
                    
                    cuts_added = self.add_cuts(P)
                    if not cuts_added:
                        break
                
                
                
                P.compute_ub()
            except InfeasibleSubproblem:
                #pruned by infeasibility
                continue
            
            if P.lb >= global_ub:
                #Pruned because of too high lb
                continue
            
            if P.ub <= global_ub:
                global_ub = P.ub
                best_solution = P.y
                #Found best global ub
            
            
            if P.lb == P.ub:
                #Pruned by optimality
                continue
            
            
            #P1,P2 = self.branch(P)
            #active_nodes.extend([P1,P2])
            del active_nodes[0]

    
    def add_cuts(self,P):
        #Add cuts
        return False

    def branch(self,P):
        pass
    

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
        
    def compute_lb(self):
        self.create_lp()
        self.solve_relaxation()
        
    def solve_relaxation(self):
        self.P.solve()
        self.lb = pulp.value(self.P.objective)

    
    def compute_ub(self):
        #Hier greedy solution of sim/ann. Schrijf waardes naar self.ub en self.y
        self.ub = 10
        self.y = [1]
        
    def create_lp(self):
        problem = self.solver.problem
        
        
        N = problem.N
        #Create LP object
        P = pulp.LpProblem("Ambulance location problem",pulp.LpMinimize)
        
        #Create variables
        x = pulp.LpVariable.dict("x",(N,N),0,1)
        y = pulp.LpVariable.dict("y",N,0,1)
            
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
    problem.loadFile('Ambulance instances/region14.txt')
    solver = Solver()
    solver.problem = problem
    
    solver.do_branch_cut()
    
