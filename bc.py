# -*- coding: utf-8 -*-
import numpy as np
import pulp
from itertools import product
from simulatedAnnealing import SAInstance
import copy 
import pickle
import time


#Todo: Infeasible error
class Solver:
    def __init__(self):
        self.verbose = True
        self.branch_cut = True
        #self.save_cuts = 'cuts.pkl'
        self.save_cuts = None
        self.load_cuts = None
        self.nodes_explored = 0
#        self.load_cuts = 'cuts.pkl'
        
    def do_branch_cut(self):
        global_ub = np.inf
        best_solution = None
        
        P = SubProblem(self.problem)
        P.create_lp()
        
        
        problem = self.problem
        
        active_nodes = [P]
        self.nodes_explored = 0
        
        if self.load_cuts is not None:
            with open(self.load_cuts,'rb') as f:
                cuts_added = pickle.load(f)
                problem.cuts_added.extend(cuts_added)
                                
                                
        while active_nodes:
            
            
            #Select active node with lowest parent lb
            parent_lbs = [P.parent_lb for P in active_nodes]
            node_key = np.argmin(parent_lbs)
            P = active_nodes[node_key]
            print('Node selected. Active nodes: {0}. GUB: {1:.3f}'.format(len(active_nodes),global_ub))
            del active_nodes[node_key]
            
            
            #check parent lb
            if P.parent_lb >= global_ub:
                print('\tParent lower bound >= GUB. Prune')
                continue
            
            
            self.nodes_explored += 1
            
            
            try:
                
                #Get lower bound, keep adding cuts until it is not possible
                first = True
                while True:
                    
                    
                    P.add_new_constraints()
                    if self.branch_cut:
                        if not first:
                            cuts_added = self.add_cuts(P)
                            problem.cuts_added.extend(cuts_added)
                            if len(cuts_added) == 0:
                                break
                            
                        
                        
                        
                        #Add cuts
                        #for N_p,M,M_p in problem.cuts_added:
                            for N_p,M,M_p in cuts_added:
                                constraint = pulp.lpSum([P.x_var[i,j] for i in N_p for j in M[i]]) + pulp.lpSum([P.x_var[i,j] for i in N_p for j in M_p]) <= problem.p + (len(N_p)-problem.p) * pulp.lpSum([P.y_var[i] for i in M_p])
                                P.P.addConstraint(constraint)
                            
                        if self.save_cuts is not None:
                            with open(self.save_cuts,'wb') as f:
                                pickle.dump(problem.cuts_added,f)
    
                    P.compute_lb()
                    
                    
                   
                    if self.branch_cut:
                        if not first:
                            print('\t{0} cut(s) added. LB:'.format(len(cuts_added),len(problem.cuts_added)),P.lb)
                        else:
                            print('\tLB {0}'.format(P.lb))
                    else:
                        #Not doing branch cut, stop immediately
                        print('\tLB {0}'.format(P.lb))
                        break

                    #P is integer or too large, stop branch and cut                    
                    if P.is_integer or P.lb >= global_ub:
                        break


                    if not first:
                        if P.lb <= prev_lb:
                            break
                    
                    first = False
                    prev_lb = P.lb
                

                #Get upper bound
                if P.is_integer:
                    #P is integer, so lb = ub
                    P.ub = P.lb
                    P.y = [i for i in problem.N if P.y_relax[i] == 1]
                
                if P.y is None:
                    P.compute_ub()
                
            except InfeasibleSubproblem:
                #pruned by infeasibility
                print('\tInfeasible subproblem')
                continue
            
            print('\tUB {1}'.format(P.lb,P.ub,global_ub))
            
            if P.lb >= global_ub:
                #Pruned because of too high lb
                print('\tToo high LB. Prune',)
                continue
            
            if P.ub < global_ub:
                global_ub = P.ub
                best_solution = P.y
                print('\tFound new global UB')
                #Found best global ub
            
            
            if P.lb == P.ub:
                #Pruned by optimality
                print('\tLB = UB')
                continue
            
            
            P1,P2 = self.branch(P)
            active_nodes.extend([P1,P2])
            print('\tBranch'.format(len(active_nodes)))
        self.solution = best_solution
        self.objective = global_ub

    
    def add_cuts(self,P):
        #Add cuts to P.P
        problem = self.problem

        cuts_added = []
        
        cut_types = [4]
        js = set()
        
        sum_of_x = np.array([sum([P.x_relax[j,i] for j in problem.N]) for i in problem.N])
        y_relax = np.array([P.y_relax[i] for i in problem.N])
        list_size = problem.p
        
        
        if 1 in cut_types:
            #select j with highest sum
            js |= set(np.argsort(-sum_of_x)[0:list_size])
            js.add(np.argmax(sum_of_x))
            #print(np.argmax(sum_of_x))
        if 2 in cut_types or 4 in cut_types:
            #select j with the largest N_p
            N_p_sizes = [0 for _ in problem.N]
            N_p_set = [set() for _ in problem.N]
            for j in problem.N:
                if P.y_relax[j] > 0:
                    for i in problem.N:
                        if P.x_relax[i,j] == P.y_relax[j]:
                            N_p_sizes[j] += 1
                            N_p_set[j].add(i)
            if 2 in cut_types:
                js |= set(np.argsort(N_p_sizes)[-list_size:])
                
                
        if 3 in cut_types:
            #select j highest sum of x / y_relax
            y_relax = np.array([P.y_relax[i] for i in problem.N])
            y_relax[np.where(y_relax == 0)] = -1 #Prevent divide by zero
            js |= set(np.argsort(sum_of_x/y_relax)[-list_size:])
            #print(np.argmax(sum_of_x/y_relax))
       
        if 4 in cut_types:
            M_sizes = [set() for _ in problem.N]
            for j in problem.N:
                for i in N_p_set[j]:
                    i_candidates = [P.x_relax[i,j] for j in problem.N]
                    i_candidates[j] = 0
                    
                    larger_0 = np.argwhere(np.array(i_candidates) > 0)
                    if len(larger_0) > 0:
                        M_sizes[j] |= set(larger_0[0])
            js |= set(np.argsort([len(ms) for ms in M_sizes])[-list_size:])
        
        for j in js:
            M = {}
            M_p = set([j]) #warehouses
            N_p = set() #clients
            

            if P.y_relax[j] == 0:
                continue
                
            for j in M_p:
                for i in problem.N:
                    if P.x_relax[i,j] == P.y_relax[j]:
                        N_p.add(i)
                        
            all_nonzeros = set()
            for i in N_p:
                x_ij = np.array([P.x_relax[i,j] for j in problem.N])
                nonzeros = set(np.argwhere(x_ij > 0)[0]) - {j}
                
                all_nonzeros |= nonzeros
                

            M = {i : set() for i in N_p}
            for j in all_nonzeros:
                x_ij = np.array([P.x_relax[i,j] for i in N_p])
                i_sorted = np.array(list(N_p))[np.argsort(-x_ij)]
                M[i_sorted[0]].add(j)


            N_p = set([i for i in M.keys()])

            M_left = set(problem.N)-( set(M_p) | set.union(*[ms for ms in M.values()]) )
            
            #print(len(M_left),len(N_p))
            all_m_nonempty = True
            for i in N_p:
                if len(M[i]) == 0:
                    try:
                        M[i].add(M_left.pop())
                    except KeyError:
                        #M is empty
                        print('Not enough in M_left')
                        all_m_nonempty = False
                        break
            else:
          
                
                
                M[list(M.keys())[0]] |= M_left
                 
                if len(N_p) > problem.p and all_m_nonempty:
                    lhs1 = np.sum([P.x_relax[j,i] for j in N_p for i in M[j]])
                    lhs2 = np.sum([P.x_relax[j,i] for j in N_p for i in M_p])
                    
                    rhs = problem.p+ (len(N_p)-problem.p) * sum([P.y_relax[i] for i in M_p])
                    #print('{0}+{1}={2} >? {3}'.format(lhs1,lhs2,lhs1+lhs2,rhs))
                    if lhs1+lhs2 > rhs:
                        #print('{0}+{1}={2} > {3}'.format(lhs1,lhs2,lhs1+lhs2,rhs))
    #                    print(N_p)
    #                    print(M)
    #                    print(M_p)
                        #cuts_added.append(pulp.LpConstraint(pulp.lpSum([P.x_var[i,j] for i in N_p for j in M[i]]) + pulp.lpSum([P.x_var[i,j] for i in N_p for j in M_p]) <= problem.p + (len(N_p)-problem.p) * pulp.lpSum([P.y_var[i] for i in M_p])))
                        cuts_added.append((N_p,M,M_p))
                    #print('len np',len(N_p),'len M', len(set.union(*[ms for ms in M.values()])), ' len mp' , len(set(M_p)))
                    
                    
 #                   stop2()
                    
                    
        #print(sum_of_x,i,sum_of_x[i])
        
        """
        
        #Select N and j
        i_p = np.argmax(sum_of_x)
        M_p = set([i_p])
        N_p = set()
        M = {}
        for i in problem.N:
        	#for all i in N, except i_p
        	if i == i_p:
        		continue
        	
        	#and except sum of x == 0
        	if sum_of_x[i] == 0:
        		continue
        
        
        	
        	#select j
        	x_ij = [P.x_relax[j,i] for j in problem.N]
        	j = np.argmax(x_ij)
        	
           # j = np.argmax(
        	N_p.add(j)
        	if j not in M:
        		M[j] = set()
        	M[j].add(i)
        	
           
        	
        M_joined = set.union(*[M[j] for j in N_p])
        M_p = M_p | (set(problem.N) - M_joined)
            
           
             
        if len(N_p) > problem.p:
            lhs1 = np.sum([P.x_relax[j,i] for j in N_p for i in M[j]])
            lhs2 = np.sum([P.x_relax[j,i] for j in N_p for i in M_p])
            
            rhs = problem.p+ (len(N_p)-problem.p) * sum([P.y_relax[i] for i in M_p])
            print('{0}+{1}={2} >? {3}'.format(lhs1,lhs2,lhs1+lhs2,rhs))
            if lhs1+lhs2 > rhs:
                print(N_p)
                print(M)
                print(M_p)
                stop2()
        #print(sum_of_x,i,sum_of_x[i])
       """
        
        return cuts_added

    def branch(self,P):
        P1 = SubProblem(self.problem)
        P1.y_one = copy.copy(P.y_one)
        P1.y_zero = copy.copy(P.y_zero)
        
        P2 = SubProblem(self.problem)
        P2.y_one = copy.copy(P.y_one)
        P2.y_zero = copy.copy(P.y_zero)
        
        #branch on closest to one
        
        
        y_branch = max(P.y_noninteger, key=P.y_noninteger.get)

        #Inherit solution if possible
        if y_branch not in P.y:
            P2.y = P.y
            P2.ub = P.ub
        else:
            P1.y = P.y
            P1.ub = P.ub
                    
        
        #set fixed y
        P1.set_y(y_branch,1)
        P2.set_y(y_branch,0)
        
        P1.dual = P.dual
        P1.reduced_cost = P.reduced_cost
        
        P2.dual = P.dual
        P2.reduced_cost = P.reduced_cost
        
        #inherit LP model
        P1.P = P.P.copy()
        P1.y_var = P.y_var
        P1.x_var = P.x_var
        
        
        P2.P = P.P.copy()
        P2.y_var = P.y_var
        P2.x_var = P.x_var
        
        #set parent lower bounds
        P1.parent_lb = P.lb
        P2.parent_lb = P.lb
        return P1,P2
       

class InfeasibleSubproblem(BaseException):
    pass

class Problem():
    def __init__(self):
        self.cuts_added = []
        
    def loadFile(self,filename):
        with open(filename,'r') as f:
            [self.n,self.p] =[int(s) for s in f.readline().split()]
            self.N = range(self.n)
            self.w = [int(s) for s in f.readline().split()]
            self.D = np.empty([self.n,self.n])
            for i in self.N:
                self.D[i] = [int(s) for s in f.readline().split()]

class SubProblem():
    def __init__(self,problem):
        self.is_integer = False
        self.problem = problem
        self.ub = None
        self.lb = None
        self.y_one = set()
        self.y_zero = set()
        
        self.y_one_new = set()
        self.y_zero_new = set()
        
        self.dual = None
        self.reduced_cost = None
        
        self.y = None
        self.parent_lb = 0
        
    def compute_lb(self):        
        self.solve_relaxation()
        
    def solve_relaxation(self):
        cplex_solver = pulp.CPLEX_PY(msg=0,mip=False)
        cplex_solver.buildSolverModel(self.P)
        logFile = 'bc.log'
        if logFile is not None:
            cplex_solver.solverModel.set_log_stream(logFile)
            cplex_solver.solverModel.set_results_stream(logFile)
            cplex_solver.solverModel.set_warning_stream(logFile)
            cplex_solver.solverModel.set_error_stream(logFile)
            
        #Add dual values from previous
        if self.dual is not None:
            cplex_solver.solverModel.start.set_start([],[],[],[],self.reduced_cost,self.dual)
            
        cplex_solver.callSolver(self.P)
        status = cplex_solver.findSolutionValues(self.P )
    
        if status == -1:
            raise InfeasibleSubproblem
        
        self.P.roundSolution()
        self.lb = pulp.value(self.P.objective)
        self.y_relax = {key:pulp.value(self.y_var[key]) for key in self.y_var}
        self.x_relax = {key:pulp.value(self.x_var[key]) for key in self.x_var}
        
        self.dual = cplex_solver.solverModel.solution.get_dual_values()
        self.reduced_cost = cplex_solver.solverModel.solution.get_reduced_costs()
      
        
        #Check if integer
        self.is_integer=True
        self.y_noninteger = {}
        for key,value in self.y_relax.items():
            if not value.is_integer():
                self.is_integer = False
                self.y_noninteger[key] = value
     
     

    
    def compute_ub(self):
        #Hier greedy solution of sim/ann. Schrijf waardes naar self.ub en self.y

        problem = self.problem
        saInstance = SAInstance(problem.N,problem.w,problem.D,problem.p)
        solution = saInstance.greedy(list(self.y_one),list(self.y_zero))
        
        self.ub = solution.objective()
        self.y = solution.solution
    

        
    def create_lp(self):
        problem = self.problem
        
        
        N = problem.N
        #Create LP object
        P = pulp.LpProblem("Ambulance location problem",pulp.LpMinimize)
        
        #Create variables
        x = pulp.LpVariable.dict("x",(N,N),0,1)
        y = pulp.LpVariable.dict("y",N,0,1)
        

            
        
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
        
        self.x_var = x
        self.y_var = y
        self.P = P
        
    def add_new_constraints(self):
        #Add zero/one constraints
        P = self.P
        for y_key in self.y_one_new:
            P += self.y_var[y_key] == 1 #float so is_integer can be called later
            
        
        for y_key in self.y_zero_new:
            P += self.y_var[y_key] == 0 #float so is_integer can be called later
    
    def set_y(self,y,value):
        if value == 0:
            self.y_one.add(y)
            self.y_one_new.add(y)
        else:
            self.y_zero.add(y)
            self.y_zero_new.add(y)            
    
    
        
if __name__ == '__main__':
    problem = Problem()
    problem.loadFile('solutions/pmed26.txt')#16 2
    #problem.p = 8
    solver = Solver()
    solver.problem = problem
    solver.branch_cut = True
    
    t = time.time()
    solver.do_branch_cut()
    print('Solution value: {0}'.format(solver.objective))
    print('Nodes explored: {0}'.format(solver.nodes_explored))
    print('Time: {0:.0f}s'.format(time.time()-t))
    print('Solution (y):',solver.solution)
    
