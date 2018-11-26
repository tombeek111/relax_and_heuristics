# -*- coding: utf-8 -*-
from ambulanceLocationInstance import AmbulanceLocationInstance
import random
import numpy as np
from itertools import product
import math

class ProbabilityAlg:
    
    def __init__(self,filename):
        ali = AmbulanceLocationInstance()
        ali.loadFile(filename)
        ali.binary = False
        ali.createLp()
        self.relaxation = None
        self.ali = ali
        
    def get_relaxation(self):
        if self.relaxation is None:
            self.ali.solve()
            
            x = self.ali.x_val
            y = self.ali.y_val
            self.relaxation = (x,y)
        return self.relaxation

        
    
    def random_solution(self,x_rel,y_rel):
        return self.random_solution_func(self,x_rel,y_rel)
        
    
    def combine_solution(self,x_sol,y_sol):
        y_sum = sum(self.y.values())
        for i,val in y_sol.items():
            if self.ali.p >= y_sum:
                return True
            
            
            if (val == 0 or self.y[i] ==0):
                
                if self.y[i] == 1:
                    y_sum -= 1
                    
                self.y[i] = 0
                
        return self.ali.p >= y_sum
    
    
    def get_x(self,y):
        D = self.ali.D.copy()
        y_zeros = [i for i,val in y.items() if val == 0]
        D[:,y_zeros] = np.inf
        n = len(self.ali.N)
        x = np.zeros((n,n))
        
        for i in self.ali.N:
            x[i,np.argmin(D[i,:])] = 1
              

        return x
    
    
    
    def get_obj(self,x):
        N = self.ali.N
        obj  = sum([x[u,v] * self.ali.w[u] * self.ali.D[u,v] for u,v in product(N,N)])
        return obj
    
    
    def get_heuristic_solution(self):
        self.y = {i : 1 for i in self.ali.N}
        
        feasible = False
        x_rel,y_rel = self.get_relaxation()
        while feasible == False:
            self.random_solution_func = round_y
            
            
            x,y = self.random_solution(x_rel,y_rel)
            
            
            feasible = self.combine_solution(x,y)
            
            
        x = alg.get_x(alg.y)
        return y,alg.get_obj(x)
    
    def get_y(self,x,N):
        N = self.ali.N
        y = {i : 0 for i in N}
        for i,j in product(N,N):
            if x[i,j] == 1:
                y[j] == 1
        return y

    
def round_y(probAlg,x_rel,y_rel):
    y = {}
    for i,val in y_rel.items():
        y[i] = 1 if random.random() <= val else 0
    return x_rel,y




def round_x(probAlg,x_rel,y_rel):
    x = {}
    for i,val in x_rel.items():
        x[i] = 1 if random.random() <= val else 0
        
    y = probAlg.get_y(x)
    return y


if __name__ == '__main__':
    funcs = {'round_x': round_x,'round_y' : round_y}
    results = {}
    
    total_iterations = 1000
    stepsize = 25
    
    for name, func in funcs.items():
        results[name] = []
        
        
        alg  = ProbabilityAlg('solutions/pmed38.txt')#38
        alg.random_solution_func = func
        for i in range(total_iterations):
            y,obj = alg.get_heuristic_solution()
            results[name].append(obj)
            if i % 100 == 0:
                print(name,i)
    
    
    for name,result in results.items():
        print('{0} : Avg {1} - Best {2}'.format(name,np.mean(result),min(result)))
        
    best_results = {}
    
    for name,result in results.items():
    
        
        best_results[name] = []
        for step in range(math.floor(total_iterations/stepsize)):
            step_result = result[step*stepsize:stepsize*(step+1)]
            best_results[name].append(min(step_result))
    
        print(name + ' best avg',np.mean(best_results[name]))
    