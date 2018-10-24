# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
import pulp
import logging

logger = logging.getLogger(__name__)

class AmbulanceLocationInstance():
    def __init__(self,name='alp'):
        self.binary = False
        self.name = name
    
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
            status = P.solve()
            
        if self.binary:
            #Round solutions to change values like 0.999999 to 1
            self.P.roundSolution()
        
        self.objective = pulp.value(self.P.objective)
        
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
        
        
        
        
    
    