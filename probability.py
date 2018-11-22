# -*- coding: utf-8 -*-
from ambulanceLocationInstance import AmbulanceLocationInstance

class ProbabilityAlg:
    def get_relaxation(self,filename):
        ali = AmbulanceLocationInstance()
        ali.loadFile(filename)
        ali.binary = False
        ali.createLp()
        ali.solve()
        
        x = ali.x_val
        y = ali.y_val
        return x,y
        pass
    
    def random_solution():
        pass
    
    def combine_solution():
        pass
    
    

if __name__ == '__main__':
    alg  = ProbabilityAlg()
    x,y = alg.get_relaxation('solutions/pmed1.txt')