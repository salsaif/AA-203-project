import numpy as np
import matplotlib.pyplot as plt
#from dubins import path_length, path_sample
import time
from scipy.spatial.distance import squareform, pdist
from utils import plot_line_segments, line_line_intersection
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

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        x_init = node(self.x_init)
        x_init.id = 0
        x_init.setP(None)
        x_init.setT(0)
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])
        Nmax = max_iters                # Number of sampled points
        V = [x_init]
        for i in range(1,Nmax + 1): # Collect sample points each node id is the same as its index in the list
            location = np.random.uniform(self.statespace_lo,self.statespace_hi)
            sample = node(location)
            sample.id = i
            V.append(sample)
        V_ids = range(0,Nmax+1)
        nodes = np.zeros((len(V),2))
        for i in range(len(V)):
            nodes[i,:] = V[i].loc
        
        D = squareform(pdist(nodes))
        w = range(1,Nmax+1) # Set W tracks nodes by id not yet added to the tree starting with all nodes in V except x_init
        H = [0] # Set H tracks the nodes by id added to the tree starting with id of x_init
        z = 0 # Variable to track the current node (z is always a node object)
        r = eps
        # Set of ids of nodes near z
        success = False
        n = 0
        while not success and n <= max_iters:
            n = n + 1 
            Hnew = []
            Nz = self.near(V, V_ids, z, r) 
            Xnear = self.intersect(Nz,w)
            for x in Xnear:
                Nx = self.near(V, V_ids,x,r)
                
                Ynear = self.intersect(Nx,H)
                    
                dists = [V[y].T+D[y,x] for y in Ynear]
                idx = np.argmin(dists)
                ymin = V[Ynear[idx]]

                if self.is_free_motion(self.obstacles, ymin.loc, V[x].loc):
                    V[x].setP(ymin)
                    V[x].setT(dists[idx])
                    Hnew.append(x)
                    w.remove(x)
            
            H = H + Hnew
            H.remove(z)
            if not H:
                success = False
                break
            
            dists = [V[y].T for y in H]
            idx = np.argmin(dists)
            z = H[idx]
            
            
            if np.linalg.norm(V[z].loc - self.x_goal) <= 0.5:
                success = True




        print n
        # print type(solution_path)
        plt.figure()
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        self.plot_tree(V, color="blue", linewidth=.5, label="RRT tree")

        goalnode = V[z]
        print goalnode.T        
        if success:

            solution_path_node = [goalnode]
            solution_path = [goalnode.loc]
            while np.all(solution_path[0] != self.x_init):
                parent = solution_path_node[0].parent
                solution_path_node = [parent] + solution_path
                solution_path = [parent.loc] + solution_path
            self.plot_path(solution_path, color="yellow", linewidth=2, label="solution path")
        plt.scatter(nodes[:,0], nodes[:,1])
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)


# Represents a geometric planning problem, where the steering solution between two points is a
# straight line (Euclidean metric)
class GeometricRRT(RRT):

    def find_nearest(self, V, x):
        # TODO: fill me in!
        num_states = len(V)
        distances = np.zeros(num_states)
        for i in range(num_states):
            distances[i] = self.dist(x,V[i])
        # print distances

        idx = np.argmin(distances)
        # print idx
        return idx
        
    def intersect(self, l1, l2):
        return list(set(l1).intersection(l2))
        
    def dist(self,x,y):
        return np.linalg.norm(np.array(x.loc)-np.array(y.loc))
        
    def near(self, V, Vid, x, r):

        dist = r

        retval = []
        for i in range(len(V)):
            if self.dist(V[x],V[i]) < dist:
                retval.append(Vid[i])

        if x in retval:
            retval.remove(x)
            
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
        plot_line_segments([(V[i].parent.loc, V[i].loc) for i in xrange(len(V)) if V[i].parent != None], **kwargs)

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
grrt.solve(1.0, 500)
elapsed = time.time() - t
print elapsed
plt.show()
