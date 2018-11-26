import numpy as np
import random
import math
import sys
import itertools
from ambulanceLocationInstance import AmbulanceLocationInstance

##% Load file
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print('Using command line file: {0}'.format(filename))
else:
    filename = './Ambulance instances/Region01.txt'
    print('Using hardcoded filename: {0}'.format(filename))
    
with open(filename,'r') as f:
    [n,p] =[int(s) for s in f.readline().split()]
    N = range(n)
    w = [int(s) for s in f.readline().split()]
    D = np.empty([n,n])
    for i in N:
        D[i] = [int(s) for s in f.readline().split()]

#Calculate objective value of a solution
def objective(solution):
    totalWeight = 0
    for i in N:
        if (i not in solution):
            minDistance = math.inf
            for j in solution:
                if D[i][j] < minDistance:
                    minDistance = D[i][j]
            totalWeight += w[i] * minDistance
    return totalWeight

#Mutate solution
def mutateSolution(solution):
    found = False
    while not found:
        newCity = random.randint(0,len(N))
        found = newCity in solution
    replaceCityIndex = random.randint(0,p - 1)
    newSolution = solution.copy()
    newSolution[replaceCityIndex] = newCity
    return newSolution


def neighborhoodIndex(solution, index):
    neighborhoodIndices = list(filter(lambda x: x not in solution, N))
    neighborhood = []
    for i in neighborhoodIndices:
        neigbhorhoodSolution = solution.copy()
        neigbhorhoodSolution[index] = i
        neighborhood.append(neigbhorhoodSolution)
    return neighborhood

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

def randomSolution():
     return random.sample(N, p) 

def mutateSolution(solution):
    replaceCityIndex = random.randint(0,len(solution) - 1)
    neighborhood = neighborhoodIndex(solution, replaceCityIndex)
    return bestCandidate(neighborhood)

def T(t, initialT):
    return (M/t)

def initialT():
    largestSum = 0
    for i in N:
        weightedSum= 0
        for j in N:
            if i != j:
                weightedSum += w[j] * D[j][i]
        if weightedSum > largestSum:
            largestSum = weightedSum
    return largestSum * 10

def acceptanceProbability(currentObjective, nextObjective, temperature):
    coefficient =  -(abs(currentObjective - nextObjective)) / temperature
    return pow(math.e,coefficient)

def greedy(p):
    solution = []
    for i in range(0,p):
        bestObjectiveCandidate = math.inf
        bestCandidate = []
        for j in range(1, len(N)):
            if j not in solution:
                candidate = solution + [j]
                objectiveCandidate = objective(candidate)
                if objectiveCandidate < bestObjectiveCandidate:
                    bestObjectiveCandidate = objectiveCandidate
                    bestCandidate = candidate
        solution = bestCandidate
    return solution

def SA(initialSolution, M):
    currentSolution = initialSolution
    bestSolution = currentSolution
    bestObjective = objective(currentSolution)
    currentObjective = bestObjective
    temperature = initialT()
    
    for t in range(1, M):
        nextSolution = mutateSolution(currentSolution)
        nextObjective = objective(nextSolution)
        #If the new solution is better, replace the current solution
        if objective(nextSolution) < objective(currentSolution):
            
            currentSolution = nextSolution
            currentObjective = nextObjective
        else:
            temperature = temperature * 0.99
            if random.random() > acceptanceProbability(currentObjective, nextObjective, temperature):
                currentSolution = nextSolution
                currentObjective = nextObjective
       
        if currentObjective < bestObjective:
            bestObjective = currentObjective
            bestSolution = currentSolution
    return bestSolution
    
M = 100
initialSolution = neighborhoodSearch(randomSolution())
print(objective(initialSolution))
print(objective(SA(initialSolution, M)))

ali = AmbulanceLocationInstance()
ali.loadFile(filename)
#ali.binary = False
ali.createLp()
ali.solve()

#print(bestObjective)
print(ali.objective)
