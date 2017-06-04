import numpy as np
import matplotlib.pyplot as plt
#from dubins import path_length, path_sample
from utils import plot_line_segments, line_line_intersection
import time
np.random.seed(1)

# Represents a motion planning problem to be solved using the RRT algorithm
class node(object):
    _id = 0
    def __init__(self, loc):
        self.loc = loc
        self._id = node._id
        node._id += 1
        self.parent = None
        self.T = None
        
    def setP(self, parent):
        self.parent = parent
        
    def setT(self, T):
        self.T = T
        
    def getid(self):
        return self._id
    def __repr__(self):
        return "%s"%(self.loc)
        
class RRT(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.n = 1
        self.ave = []
    # Subject to the robot dynamics, returns whether a point robot moving along the shortest
    # path from x1 to x2 would collide with any obstacles (implemented for you as a "black box")
    # INPUT: (obstacles, x1, x2)
    #   obstacles - list/np.array of line segments ("walls")
    #          x1 - start state of motion
    #          x2 - end state of motion
    # OUTPUT: Boolean True/False
    def is_free_motion(self, obstacles, x1, x2):
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    # Given a list of states V and a query state x, returns the index (row) of V such that the
    # steering distance (subject to robot dynamics) from V[i] to x is minimized
    # INPUT: (obstacles, x1, x2)
    #           V - list/np.array of states ("samples")
    #           x - query state
    # OUTPUT: Integer index of nearest point in V to x
    def find_nearest(self, V, x):
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    # Steers from x towards y along the shortest path (subject to robot dynamics); returns y if
    # the length of this shortest path is less than eps, otherwise returns the point at distance
    # eps along the path from x to y.
    # INPUT: (obstacles, x1, x2)
    #           x - start state
    #           y - target state
    #         eps - maximum steering distance
    # OUTPUT: State (numpy vector) resulting from bounded steering
    def steer_towards(self, x, y, eps):
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    # Constructs an RRT rooted at self.x_init with the aim of producing a dynamically-feasible
    # and obstacle-free trajectory from self.x_init to self.x_goal.
    # INPUT: (eps, max_iters, goal_bias):
    #         eps - maximum steering distance
    #   max_iters - maximum number of RRT iterations (early termination is possible when a feasible
    #               solution is found)
    #   goal_bias - probability during each iteration of setting x_rand = self.x_goal
    #               (instead of uniformly randomly sampling from the state space)
    # OUTPUT: None officially (just plots), but see the "Intermediate Outputs" descriptions below
    def solve(self, eps, max_iters = 1000, goal_bias = 0.05):
        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        x_init = node(self.x_init)
        x_init.setP(None)
        x_init.setT(0)
        V = [x_init]
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - succcess: whether or not you've found a solution within max_iters RRT iterations
        #    - solution_path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        # TODO: fill me in!
        success = False
        i = 0
        while not success and i <= max_iters:
            i = i + 1
            z = np.random.uniform()
            if z < goal_bias :
                x_rand = self.x_goal
            else:
                x_rand = np.random.uniform(self.statespace_lo, self.statespace_hi)
            success, V = self.extend(V,x_rand,eps)



        self.plot(V,success)


# Represents a geometric planning problem, where the steering solution between two points is a
# straight line (Euclidean metric)
class GeometricRRT(RRT):
    
    def extend(self,V,x_rand,eps):
        success = False
        x_near = V[self.find_nearest(V, x_rand)]
        x_new = node(self.steer_towards(x_near.loc, x_rand, eps))
        

        if self.is_free_motion(self.obstacles, x_near.loc, x_new.loc):
            V.append(x_new)
            x_new.setT(x_near.T + np.linalg.norm(np.array(x_new.loc)-np.array(x_near.loc)))
            cmin = x_new.T
            zmin = x_near
            Z = self.near(V,x_new, eps)
            self.ave.append(float(len(Z))/len(V))

            if Z:
                for i in range(len(Z)):
                    z_new = self.steer_towards(Z[i].loc,x_new.loc,eps)
                    dist = np.linalg.norm(np.array(z_new)-np.array(Z[i].loc))
                    if self.is_free_motion(self.obstacles, Z[i].loc, z_new) and np.all(z_new == x_new.loc) and Z[i].T+dist < cmin:
                        #print "improve parent"
                        cmin = Z[i].T + dist
                        zmin = Z[i]
                        
            x_new.setT(zmin.T + np.linalg.norm(np.array(x_new.loc)-np.array(zmin.loc)))
            x_new.setP(zmin)
            
                      
            if Z:
                if zmin in Z:
                    Z.remove(zmin)
                for i in range(len(Z)):
                    z_near = self.steer_towards(x_new.loc,Z[i].loc,eps)
                    if np.all(z_near == Z[i].loc):
                        dist = np.linalg.norm(np.array(x_new.loc)-np.array(Z[i].loc))
                        T_near = Z[i].T

                        if self.is_free_motion(self.obstacles, x_new.loc, Z[i].loc) and x_new.T+dist < T_near:
                            Z[i].setP(x_new)
                    
            self.n = self.n+1
            if np.all(x_new.loc == self.x_goal):
                success = True
        
        return success, V
        

    def find_nearest(self, V, x):
        # TODO: fill me in!
        num_states = len(V)
        distances = np.zeros(num_states)
        for i in range(num_states):
            distances[i] = np.linalg.norm(np.array(x)-np.array(V[i].loc))
        # print distances

        idx = np.argmin(distances)
        # print idx
        return idx

    def plot(self,V,success):
        plt.figure()
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        self.plot_tree(V, color="blue", linewidth=.5, label="RRT tree")
        nodes = np.zeros((self.n,2))
        for i in range(len(V)):
            nodes[i,:] = V[i].loc
            if np.all(V[i].loc == self.x_goal):
                goalnode = V[i]
        if success:

            solution_path_node = [goalnode]
            solution_path = [goalnode.loc]
            while np.all(solution_path[0] != self.x_init):
                parent = solution_path_node[0].parent
                solution_path_node = [parent] + solution_path
                solution_path = [parent.loc] + solution_path
            self.plot_path(solution_path, color="green", linewidth=2, label="solution path")
            print np.mean(self.ave)
            print goalnode.T
        plt.scatter(nodes[:,0], nodes[:,1])
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        
    def steer_towards(self, x, y, eps):
        dist = np.linalg.norm(np.array(x)-np.array(y))
        if dist < eps:
            return y
        return x + eps*(y-x)/dist
        
    def near(self, V, x, eps):
        dist = min(np.sqrt(100*np.log(self.n)/(np.pi*self.n)), eps)

        retval = []
        for i in range(self.n):
            if np.linalg.norm(np.array(x.loc)-np.array(V[i].loc)) <= dist:
                retval.append(V[i])

            
        return retval


    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        # print "motion = ", motion
        for line in obstacles:
            if line_line_intersection(motion, line):
                # print "False"
                return False
        # print "True"
        return True

    def plot_tree(self, V, **kwargs):
        plot_line_segments([(V[i].parent.loc, V[i].loc) for i in range(len(V)) if V[i].parent != None], **kwargs)

    def plot_path(self, path, **kwargs):
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], **kwargs)




### TESTING

MAZE = np.array([
    ((5, 5), (-5, 5)),
    ((-5, 5), (-5, -5)),
    ((-5,-5), (5, -5)),
    ((5, -5), (5, 5)),
    ((-3, -3), (-3, -1)),
    ((-3, -3), (-1, -3)),
    ((3, 3), (3, 1)),
    ((3, 3), (1, 3)),
    ((1, -1), (3, -1)),
    ((3, -1), (3, -3)),
    ((-1, 1), (-3, 1)),
    ((-3, 1), (-3, 3)),
    ((-1, -1), (1, -3)),
    ((-1, 5), (-1, 2)),
    ((0, 0), (1, 1))
])
t = time.time()
grrt = GeometricRRT([-5,-5], [5,5], [-4,-4], [4,4], MAZE)
grrt.solve(1.0, 2000)
elapsed = time.time() - t
print elapsed
plt.show()
