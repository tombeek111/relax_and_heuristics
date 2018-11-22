# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
import pulp
import logging
import math
import time

logger = logging.getLogger(__name__)

class AmbulanceLocationInstance():
    def __init__(self,name='alp'):
        self.binary = False
        self.name = name
        self.additionalCuts = []
    
    def loadFile(self,filename):
        with open(filename,'r') as f:
            [self.n,self.p] =[int(s) for s in f.readline().split()]
            self.N = range(self.n)
            self.w = [int(s) for s in f.readline().split()]
            self.D = np.empty([self.n,self.n])
            for i in self.N:
                self.D[i] = [int(s) for s in f.readline().split()]
    
    def createLp(self):
        N = self.N
        #Create LP object
        P = pulp.LpProblem("Ambulance location problem",pulp.LpMinimize)
        
        #Create variables
        if self.binary:
            x = pulp.LpVariable.dict("x",(N,N),0,1,pulp.LpBinary)
            y = pulp.LpVariable.dict("y",N,0,1,pulp.LpBinary)
        else:
            x = pulp.LpVariable.dict("x",(N,N),0,1)
            y = pulp.LpVariable.dict("y",N,0,1)
            
        self.x_var = x
        self.y_var = y
        #set cost function
        P += pulp.lpSum(x[u,v] * self.w[u] * self.D[u,v] for u,v in product(N,N))
        
        #Add constraints
        #serve each city
        for u in self.N:
            P += pulp.lpSum(x[u,v] for v in N) == 1
            
        #Only select an edge(u,v) if there is an ambulance post at v
        for u,v in product(N,N):
            P += x[u,v] <= y[v]
        
        #Max p ambulance posts
        P += pulp.lpSum(y[u] for u in N) <= self.p
        
        if 'dominated_edges' in self.additionalCuts:
            #Hier cut van marcel
            pass
        
        if 'partitions' in self.additionalCuts:
            #hier cut van Tom
            p1 = 1.2
            p2 = 0.5
            
            N_p = set(np.argsort(self.w)[::-1][:math.ceil(self.p*p1)])
            N_p.pop()
            M_j = {}
            weights = np.matrix([[self.w[u]*self.D[u,v] for v in N] for u in N])
            weights[list(set(N)-set(N_p)),:] = np.inf
            
            #print(weights)
          
                      
            for i,n_p in enumerate(N_p):
                
                min_u = np.where(np.argmin(weights,axis=0) == n_p)[1]
                u_values = [weights[n_p,u] for u in min_u]
                u_argsort = np.argsort(u_values)
                u_selected = min_u[u_argsort[0:math.ceil(p2*len(min_u))]]
     
                #min_u = min_u[0:math.ceil(0.25*len(min_u))]
                M_j[n_p] = set(u_selected)
             
            
            M_p = set(N)-set.union(*[m_j for m_j in M_j.values()])
            
            if True:
                P += (pulp.lpSum([x[u,v] for u in N_p for v in M_j[u]]) +
                pulp.lpSum(x[u,v] for (u,v) in product(N_p,M_p)) <=
                self.p + (len(N_p)-self.p) * pulp.lpSum(y[u] for u in M_p))
            
            
            P += pulp.lpSum(x[u,v] for u,v in product(N,N) if u != v) == len(N)-self.p
            
            P += pulp.lpSum(x[u,u] for u in N) == self.p
            
            weights = np.matrix([[self.w[u]*self.D[u,v] for v in N] for u in N])
            for u in N:
                weights[u,u] = np.inf
                
            second_best = np.amin(weights,axis=1).flatten()
   
            lower_bound = np.sum(np.sort(second_best).tolist()[0][0:len(N)-self.p])
            
            P += pulp.lpSum(y[u] for u in N) == self.p
            P += pulp.lpSum(x[u,v] * self.w[u] * self.D[u,v] for u,v in product(N,N)) >= lower_bound
            #2
            #for u,v in product(N,N):
            #    P += x[u,v]+x[v,u] <= 1
        
        self.P = P
        return P


    def solve(self):
        P = self.P
        cplex_solver = pulp.CPLEX_PY(msg=0)
        if cplex_solver.available():
            cplex_solver.buildSolverModel(P)
            
            logFile = 'solver_{0}.log'.format(self.name)
            cplex_solver.solverModel.set_log_stream(logFile)
            cplex_solver.solverModel.set_results_stream(logFile)
            cplex_solver.solverModel.set_warning_stream(logFile)
            cplex_solver.solverModel.set_error_stream(logFile)
        
        
            cplex_solver.callSolver(P)
            logger.info('Solving by cplex. Logfile: ' + logFile)
            status = cplex_solver.findSolutionValues(P)
            
            self.duration = cplex_solver.solverModel.get_dettime()
            self.nodes = cplex_solver.solverModel.solution.progress.get_num_nodes_processed()
            #P.setSolver(cplex_solver)
        else:
            
            logger.info('Solving by pulp standard solver')
            startTime = time.time()
            status = P.solve()
            self.duration = time.time()-startTime
            self.nodes = 0
            
        if self.binary:
            #Round solutions to change values like 0.999999 to 1
            self.P.roundSolution()
        
        
        self.objective = pulp.value(self.P.objective)
        self.y_val = {i : self.y_var[i].varValue for i in self.N}
        self.x_val = {(i,j) : self.x_var[(i,j)].varValue for i,j in product(self.N,self.N)}
       
        #print('{0} problem objective value: {1}'.format(self.name,self.objective))
        #print('Log file: {0}'.format(logFile))
    
    
        return status
    
    def solveFromFile(filename,name='alp',binary=True,additionalCuts=[]):
        alp = AmbulanceLocationInstance(name)
        alp.binary=binary
        alp.additionalCuts = additionalCuts
        
        alp.loadFile(filename)
        alp.createLp()
        alp.solve()
        return alp
        
        
        
        
    
    