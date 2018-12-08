import numpy as np
import random
import math
import sys
import itertools

from ambulanceLocationInstance import AmbulanceLocationInstance
#from bc import SubProblem

###% Load file
#if len(sys.argv) > 1:
#    filename = sys.argv[1]
#    print('Using command line file: {0}'.format(filename))
#else:
#    filename = './Ambulance instances/Region01.txt'
#    print('Using hardcoded filename: {0}'.format(filename))
    
#with open(filename,'r') as f:
#    [n,p] =[int(s) for s in f.readline().split()]
#    N = range(n)
#    w = [int(s) for s in f.readline().split()]
#    D = np.empty([n,n])
#    for i in N:
#        D[i] = [int(s) for s in f.readline().split()]

class SAInstance():
    def __init__(self, N, w, D, p):
        self.N = N
        self.w = w
        self.D = D
        self.p = p
    
    def loadFile(self,filename):
        with open(filename,'r') as f:
            [self.n,self.p] =[int(s) for s in f.readline().split()]
            self.N = range(self.n)
            self.w = [int(s) for s in f.readline().split()]
            self.D = np.empty([self.n,self.n])
            for i in self.N:
                self.D[i] = [int(s) for s in f.readline().split()]

    def randomSolution(self, y_one = [], y_zero = []):
        sampleRange = list(set(self.N) - set(y_zero) - set(y_one))
        sample = random.sample(sampleRange, self.p)
        sample[:len(y_one)] = y_one
        return SASolution(sample, self)

    def greedy(self, y_one = [], y_zero = []):
        solution = SASolution(y_one, self)
        for i in range(0,self.p - len(y_one)):
            bestObjectiveCandidate = math.inf
            bestCandidate = SASolution([], self)
            for j in range(1, len(self.N)):
                if ((j not in solution.solution) and (j not in y_zero)):
                    candidate = SASolution(solution.solution + [j], self)
#                    candidate = solution + [j]
                    objectiveCandidate = candidate.objective()
                    if objectiveCandidate < bestObjectiveCandidate:
                        bestObjectiveCandidate = objectiveCandidate
                        bestCandidate = candidate
            solution = bestCandidate
        return solution

    def SA(self, initialSolution, M,stopafter=100, y_one=[], y_zero=[]):
        currentSolution = initialSolution
        bestSolution = currentSolution
        bestObjective = currentSolution.objective()
        currentObjective = bestObjective
        temperature = self.initialT()
        lastAccept = 0
        
        
        for t in range(1, M):
            if t-lastAccept > stopafter:
                break
            
            nextSolution = currentSolution.mutateSolution(y_one,y_zero)
            nextObjective = nextSolution.objective()
            #If the new solution is better, replace the current solution
            if nextObjective < currentObjective:
                
                currentSolution = nextSolution
                currentObjective = nextObjective
                lastAccept = t
            else:
                temperature = temperature * 0.9
                accProb = acceptanceProbability(currentObjective, nextObjective, temperature)
#                print(temperature,accProb)
                if random.random() <= accProb:
                    currentSolution = nextSolution
                    currentObjective = nextObjective
                    lastAccept = t
           
            if currentObjective < bestObjective:
                bestObjective = currentObjective
                bestSolution = currentSolution
        return bestSolution
    
    def initialT(self):
        largestSum = 0
        for i in self.N:
            weightedSum= 0
            for j in self.N:
                if i != j:
                    weightedSum += self.w[j] * self.D[j][i]
            if weightedSum > largestSum:
                largestSum = weightedSum
        return largestSum * 2.5


class SASolution():
    def __init__(self, solution, instance):
        self.solution = solution
        self.N = instance.N
        self.w = instance.w
        self.D = instance.D
        self.p = instance.p
        self.instance = instance

    #Calculate objective value of a solution
    def objective(self):
        totalWeight = 0
        for i in self.N:
            if (i not in self.solution):
                minDistance = math.inf
                for j in self.solution:
                    if self.D[i][j] < minDistance:
                        minDistance = self.D[i][j]
                totalWeight += self.w[i] * minDistance
        return totalWeight

    #Mutate solution
    def mutateSolution(self, y_one=[], y_zero=[]):
        
        found = False
        while not found:
            newCity = random.randint(0,len(self.N) -1)
            found = newCity not in self.solution and newCity not in y_zero
        replaceCityIndex = random.randint(len(y_one),self.p - 1)
        newSolution = self.solution.copy()
        newSolution[replaceCityIndex] = newCity
        return SASolution(newSolution, self.instance)


    def neighborhoodIndex(self,solution, index):
        neighborhoodIndices = list(filter(lambda x: x not in solution.solution, self.N))
        neighborhood = []
        for i in neighborhoodIndices:
            neigbhorhoodSolution = solution.solution.copy()
            neigbhorhoodSolution[index] = i
            neighborhood.append(neigbhorhoodSolution)
        return SASolution(neighborhood, self.instance)

def neighborhood(solution):
    neighborhood = map(lambda x: neighborhoodIndex(solution,x), range(0,len(solution)))   
    return list(itertools.chain.from_iterable(neighborhood))

    
def bestCandidate(solutions):
    objectiveCandidates = list(map(lambda x: objective(x), solutions))
    bestCandidate = objectiveCandidates.index(min(objectiveCandidates))
    return solutions[bestCandidate]

def neighborhoodSearch(solution):
    neighborhoodList = neighborhood(solution)
    bestSolution = bestCandidate(neighborhoodList)
    if objective(bestSolution) < objective(solution):
        return neighborhoodSearch(bestSolution)
    return solution


#def mutateSolution(solution):
#    replaceCityIndex = random.randint(0,len(solution.solution) - 1)
#    neighborhood = neighborhoodIndex(solution, replaceCityIndex)
#    return bestCandidate(neighborhood)
#
#def T(t, initialT):
#    return (M/t)


def acceptanceProbability(currentObjective, nextObjective, temperature):
    coefficient =  -(abs(currentObjective - nextObjective)) / temperature
    return pow(math.e,coefficient)

#def greedy(p):
#    solution = []
#    for i in range(0,p):
#        bestObjectiveCandidate = math.inf
#        bestCandidate = []
#        for j in range(1, len(N)):
#            if j not in solution:
#                candidate = solution + [j]
#                objectiveCandidate = objective(candidate)
#                if objectiveCandidate < bestObjectiveCandidate:
#                    bestObjectiveCandidate = objectiveCandidate
#                    bestCandidate = candidate
#        solution = bestCandidate
#    return solution



#def compute_ub_sa(problem):
#    N = problem.solver.problem.N
#    w = problem.solver.problem.w
#    D = problem.solver.problem.D
#    p = problem.solver.problem.p
#    instance = SAInstance(N,w,D,p)
#    solution = instance.randomSolution([1,3],[0,4])
#    solution = instance.greedy()
#    print(problem.y_one)
#    print(problem.y_zero)
#    return solution.objective()
#
#M = 100
#initialSolution = neighborhoodSearch(randomSolution())
#print(objective(initialSolution))
#print(objective(SA(initialSolution, M)))
#
#ali = AmbulanceLocationInstance()
#ali.loadFile(filename)
##ali.binary = False
#ali.createLp()
#ali.solve()
#
##print(bestObjective)
#print(ali.objective)
