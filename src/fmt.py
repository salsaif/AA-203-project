import numpy as np
import matplotlib.pyplot as plt
#from dubins import path_length, path_sample
from utils import plot_line_segments, line_line_intersection
np.random.seed(89)

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
        x_init.setP(None)
        x_init.setT(0)
        H = [x_init]

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
        r = eps
        x_rand = np.random.uniform(self.statespace_lo, self.statespace_hi,(max_iters,2))
        V = [x_init]
        w = []
        for row in x_rand:
            w.append(node(row))
        V = V + w
        z = x_init
        
        while not success and i <= max_iters:
            i = i + 1 
            print i          
            Hnew = []
            Nz = self.near(V,z,r)
            Xnear = self.intersect(Nz,w)
            for x in Xnear:
                Nx = self.near(V,x,r)
                
                Ynear = self.intersect(Nx,H)
                if not Ynear:
                    print len(Nx)
                    print len(H)
                dists = [y.T for y in Ynear]
                idx = np.argmin(dists)
                ymin = Ynear[idx]

                if self.is_free_motion(self.obstacles, ymin.loc, x.loc):
                    x.setP(ymin)
                    x.setT(ymin.T + self.dist(ymin,x))
                    Hnew.append(x)
                    w.remove(x)
            
            H = H + Hnew
            H.remove(z)
            if not H:
                success = False
                break
            
            dists = [y.T for y in H]
            idx = np.argmin(dists)
            z = H[idx]
            
            
            if np.all(z.loc == self.x_goal):
                success = True




        # print P
        # print type(solution_path)
        plt.figure()
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        self.plot_tree(V, color="blue", linewidth=.5, label="RRT tree")
        nodes = np.zeros((len(V),2))
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
        #print goalnode.T
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
        
    def near(self, V, x, r):

        dist = r

        retval = []
        for i in range(len(V)):
            if self.dist(x,V[i]) < dist:
                retval.append(V[i])

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

grrt = GeometricRRT([-5,-5], [5,5], [-4,-4], [4,4], MAZE)
grrt.solve(3.0, 2000)

#drrt = DubinsRRT([-5,-5,0], [5,5,2*np.pi], [-4,-4,0], [4,4,np.pi/2], MAZE, .5)
#drrt.solve(3.0, 1000)

plt.show()
