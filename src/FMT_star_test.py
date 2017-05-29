import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection
np.random.seed(1)

# Class node that defines a node object
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
        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        x_init = node(self.x_init)
        x_init.id = 0
        x_init.setP(None)
        x_init.setT(0)
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])
        Nmax = 5                # Number of sampled points
        V_nodes = [x_init]
        for i in range(1,Nmax + 1): # Collect sample points each node id is the same as its index in the list
            location = np.random.uniform(self.statespace_lo,self.statespace_hi,size = (1,x_init.loc.size))
            sample = node(location)
            sample.id = i
            V_nodes.append(sample)
        V_ids = set(range(0,Nmax+1))
        W = set(range(1,Nmax+1)) # Set W tracks nodes by id not yet added to the tree starting with all nodes in V except x_init
        H = {0} # Set H tracks the nodes by id added to the tree starting with id of x_init
        z = x_init # Variable to track the current node (z is always a node)
        Nz = near(V_nodes, V_ids, z, 3.0) # Set of ids of nodes near z
        
        # TODO: fill me in!
        success = False
        i = 0
        while not success and i <= max_iters:
            i = i + 1
            if z.loc == x_goal:
                success = True
                break
            H_new = set()
            X_near = Nz & W # Set of ids of nodes not in the tree but near z
            for x in X_near:
                Nx = near(V_nodes, V_ids, V_nodes[x], 3.0) # Set of ids of nodes near x
                Y_near = Nx & H  # Set of ids of nodes near x and in the tree
                tmp1 = [] # temporary empty list
                tmp2 = [] # temporary empty list
                for y in Y_near: # Iterate through the set of ids in Y_near
                    tmp1.append(LA.norm(x_init.loc-V_nodes[y].loc) + LA.norm(V_nodes[y].loc-V_nodes[x].loc)) # add the cost of each node to the list tmp1
                    tmp2.append(y)
                tmp_idx = np.argmin(tmp1) # index of argmin in the list tmp1
                idx_min = tmp2[tmp_idx] # find the id of the node 
                y_min = V_nodes[idx_min]
                if is_free_motion(obstacles, y_min.loc, V_nodes[x].loc):
                    # add edge to the tree
                    H_new = H_new | set(x)
                    W = W - set(x)
            H = H | H_new - set(z.id)
            if H == set():
                success = False
                break
            tmp1 = []
            tmp2 = []
            for y in H: # Iterate through the set of ids in Y_near
                tmp1.append(LA.norm(x_init.loc-V_nodes[y].loc)) # add the cost of each node to the list tmp1
                tmp2.append(y)
            tmp_idx = np.argmin(tmp1) # index of argmin in the list tmp1
            idx_min = tmp2[tmp_idx] # find the id of the node 
            z = V_nodes[idx_min] # update the value of z

        # print P
        # print type(solution_path)
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

        print goalnode.T
        plt.scatter(nodes[:,0], nodes[:,1])
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)


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
            Z = self.near(V,x_new)
            #if zmin not in Z:
            #    print "n = ", self.n
            #    print "V = ",len(V)
            #    print "Z = ",len(Z)
            #    print "dist = ", np.sqrt(100*np.log(self.n)/(np.pi*self.n))
            #    print "x = ",x_new.loc
            #    print "zmin = ", zmin.loc
            #    for z in Z:
            #        print "z = ",z.loc
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
                        #print "T_near = ", T_near
                        #print "comp = ", T[self.n]+dist
                        if self.is_free_motion(self.obstacles, x_new.loc, Z[i].loc) and x_new.T+dist < T_near:
                            #print "rewiring"
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


    def steer_towards(self, x, y, eps):
        # TODO: fill me in!
        dist = np.linalg.norm(np.array(x) - np.array(y))
        if dist < eps:
            return y
        dx = x[0] - y[0]
        dy = x[1] - y[1]
        cos = dx/dist
        sin = dy/dist
        return np.array([x[0]+eps*cos, x[1]+eps*sin])


    # Takes in a list of nodes, set of node ids, specific point, and distance
    # and return a set of node ids of the nodes within the distance 
    def near(self, V_nodes, V_ids, x, r_n = 3.0):
        dist = r_n
        retval = []
        retset = set()
        for i in V_ids - set(x.id):
            if np.linalg.norm(np.array(x.loc)-np.array(V_nodes[i].loc)) <= dist:
                retval.append(V_nodes[i])
                retset = retset | set(V_nodes[i].id)            
        return retset


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

grrt = GeometricRRT([-5,-5], [5,5], [-4,-4], [4,4], MAZE)
grrt.solve(3.0, 2000)
plt.show()
