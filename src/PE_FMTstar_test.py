import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import squareform, pdist
from utils import plot_line_segments, line_line_intersection
import copy
np.random.seed(1)

# This object contains the location of the node, its parent and its cost (T). Every node has its own unique id.
class node(object):
    _id = 0
    def __init__(self, loc):
        self.loc = loc
        self._id = node._id
        node._id += 1
        self.parent = None
        self.children = []
        self.T = None
        self.numchild = 0
        
    def setP(self, parent):
        self.parent = parent

    def addChild (self, child):
        self.children.append(child)
        self.numchild += 1
        
    def setT(self, T):
        self.T = T
        
    def getid(self):
        return self._id
        
    def __repr__(self):
        return "%s"%(self.loc)
        
    def remove_child(self, child):
        if child in self.children:
            self.numchild -= 1
            self.children.remove(child)
        
# Represents a motion planning problem to be solved using the FMT algorithm

def isBetween(obstacles, node, margin):
        vertix_cnt = 0
        for line in obstacles:
            if vertix_cnt >= 4 and vertix_cnt%4 == 0: # Skip first rectangle which defines the state space walls
                # initialize the x and y min/max variables using the first side in the obstacle 
                pt1, pt2 = line 
                x_min = min(pt1[0], pt2[0])
                y_min = min(pt1[1], pt2[1])
                x_max = max(pt1[0], pt2[0])
                y_max = max(pt1[1], pt2[1])
                
                # loop through all sides of the obstacle to find the diagonal points (lower left and upper right corners)
                for side in obstacles[vertix_cnt:vertix_cnt+4]:
                    a, b = np.array(side)
                    x_min = min(a[0],b[0], x_min)
                    x_max = max(a[0],b[0], x_max)
                    y_min = min(a[1],b[1], y_min)
                    y_max = max(a[1],b[1], y_max)

                # check if node is inside the rectangle
                if node[0] <= x_max + margin and node[0] >= x_min - margin:
                    if node[1] <= y_max + margin and node[1] >= y_min - margin:
                        return True
            else:
                a, b = np.array(line)
                if node[0] <= max(a[0],b[0]) + margin and node[0] >= min(a[0],b[0]) - margin:
                    if node[1] <= max(a[1],b[1]) + margin and node[1] >= min(a[1],b[1]) - margin:
                        return True
            vertix_cnt = vertix_cnt + 1
        return False

class FMT(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.cost = None
        self.V = []

    # Subject to the robot dynamics, returns whether a point robot moving along the shortest
    # path from x1 to x2 would collide with any obstacles (implemented for you as a "black box")
    # INPUT: (obstacles, x1, x2)
    #   obstacles - list/np.array of line segments ("walls")
    #          x1 - start state of motion
    #          x2 - end state of motion
    # OUTPUT: Boolean True/False
    
    def is_free_motion(self, obstacles, x1, x2):
        raise NotImplementedError("is_free_motion must be overriden by a subclass of FMT")

    # Given a list of states V and a query state x, returns the index (row) of V such that the
    # steering distance (subject to robot dynamics) from V[i] to x is minimized
    # INPUT: (obstacles, x1, x2)
    #           V - list/np.array of states ("samples")
    #           x - query state
    # OUTPUT: Integer index of nearest point in V to x
    def find_nearest(self, V, x):
        raise NotImplementedError("find_nearest must be overriden by a subclass of FMT")


    # Constructs an FMT rooted at self.x_init with the aim of producing a dynamically-feasible
    # and obstacle-free trajectory from self.x_init to self.x_goal.
    # INPUT: (r, max_iters, goal_bias):
    #         r - maximum steering distance
    #   max_iters - maximum number of FMT iterations (early termination is possible when a feasible
    #               solution is found)

    def solve(self, r, sample, max_iters = 1000, sp = 1.0, pursuer = False):

        x_init = node(self.x_init)
        x_init.id = 0
        x_init.setP(None)
        x_init.setT(0)
        Nmax = max_iters                # Number of sampled points
        self.V = [x_init] + sample
        V_ids = range(0,len(self.V))
        nodes = np.zeros((len(self.V),2))

        for i in range(len(self.V)):
            nodes[i,:] = self.V[i].loc
        D = squareform(pdist(nodes))
        w = range(1,len(self.V)) # Set W tracks nodes by id not yet added to the tree starting with all nodes in V except x_init
        H = [0] # Set H tracks the nodes by id added to the tree starting with id of x_init
        z = 0 # Variable to track the current node (z is always a node object)
        n = 0
        while n <= max_iters:
            n = n + 1 
            Hnew = []
            Xnear = self.near(w, z, r) 
            for x in Xnear:
                Ynear = self.near(H,x,r)
                dists = [self.V[y].T+sp*D[y,x] for y in Ynear]
                idx = np.argmin(dists)
                ymin = self.V[Ynear[idx]]
                xnew = self.V[x]
                if self.is_free_motion(self.obstacles, ymin.loc, xnew.loc):
                    ymin.addChild(xnew)
                    xnew.setP(ymin)
                    xnew.setT(dists[idx])
                    Hnew.append(x)
                    w.remove(x)
            
            H = H + Hnew

            H.remove(z)
            if not H:
                break
            
            dists = [self.V[y].T for y in H]
            idx = np.argmin(dists)
            z = H[idx]
            

            



# Represents a geometric planning problem, where the steering solution between two points is a
# straight line (Euclidean metric)
class GeometricFMT(FMT):
    
    def isBetween(self, obstacles, node, margin):
        for line in obstacles:
            a, b = np.array(line)
            if node[0] <= max(a[0],b[0]) + margin and node[0] >= min(a[0],b[0]) - margin:
                if node[1] <= max(a[1],b[1]) + margin and node[1] >= min(a[1],b[1]) - margin:
                    return True
        return False
    
        
    def dist(self,x,y):
        return np.linalg.norm(np.array(x.loc)-np.array(y.loc))
        
    def near(self, Vid, x, r):
        n = len(self.V)
        dist = min(np.sqrt(100*np.log(n)/(np.pi*n)),r)

        retval = []
        for i in Vid:
            if self.dist(self.V[x],self.V[i]) < dist:
                retval.append(i)

        if x in retval:
            retval.remove(x)
            
        return retval

    def remove_branch(self, node):
        n = len(node.children)
        i = n-1
        while i >= 0:
            child = node.children[i]
            if child.numchild == 0:
                try:
                    self.V.remove(child)
                except ValueError:
                    print("node wasn't in V (1)...")
            else:
                self.remove_branch(child)
            
            i -= 1

        try: 
            self.V.remove(node)
            if node.parent is not None:
                node.parent.remove_child(node)
        except ValueError:
            print("node wasn't in V (3)...")
        return

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self,**kwargs):
        plot_line_segments([(self.V[i].parent.loc, self.V[i].loc) for i in xrange(len(self.V)) if self.V[i].parent != None], **kwargs)
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")


    def plot_path(self, path, **kwargs):
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], **kwargs)
        
    def plot_all(self, goal = None):
        plt.figure()
        nodes = np.zeros((len(self.V),2))
        if goal is None:
            goal = self.x_goal
        success = False
        for i in range(len(self.V)):
            nodes[i,:] = self.V[i].loc
            if np.linalg.norm(self.V[i].loc - goal,np.inf) <= 0.5:
                success = True
                goalnode = self.V[i]
            
        self.plot_tree(color="blue", linewidth=.5, label="FMT tree")
        plt.scatter(nodes[:,0], nodes[:,1])
        xgoal = np.linspace(goal[0]-0.5,goal[0]+0.5,10)
        ygoal1 = (goal[1]-0.5)*np.ones(10)
        ygoal2 = (goal[1]+0.5)*np.ones(10)
        plt.fill_between(xgoal,ygoal1,ygoal2,color="green")
        plt.scatter([self.x_init[0]], [self.x_init[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        
        
        
        if success:
            self.cost = goalnode.T
            solution_path_node = [goalnode]
            solution_path = [goalnode.loc]
            while np.all(solution_path[0] != self.x_init):
                parent = solution_path_node[0].parent
                solution_path_node = [parent] + solution_path
                solution_path = [parent.loc] + solution_path
            self.plot_path(solution_path, color="green", linewidth=5, label="solution path")





### TESTING

MAZE = np.array([
    ((5, 5), (-5, 5)),
    ((-5, 5), (-5, -5)),
    ((-5,-5), (5, -5)),
    ((5, -5), (5, 5)),
    ((-3, -3), (-3, -1)),
    ((-3, -1), (-1, -1)),
    ((-1, -1), (-1, -3)),
    ((-1, -3), (-3, -3)),
    ((3, 3), (3, 1)),
    ((3, 1), (1, 1)),
    ((1, 1), (1, 3)),
    ((1, 3), (3, 3)),
    ((1, -1), (3, -1)),
    ((3, -1), (3, -3)),
    ((3, -3), (1, -3)),
    ((1, -3), (1, -1)),
#    ((-1, 1), (-3, 1)),
#    ((-3, 1), (-3, 3)),
#    ((-1, -1), (1, -3)),
#    ((-1, 5), (-1, 2)),
#    ((0, 0), (1, 1))
])
#t = time.time()
#grrt = GeometricFMT([-5,-5], [5,5], [-4,-4], [4,4], MAZE)
#grrt.solve(1.0, 500)
#elapsed = time.time() - t
#print "Elapsed time = ", elapsed
#plt.show()

####################### TESTING PE WITH FMT* ################################

# 1) Create uniform sample of nodes to be used by both agents to build their 
#    respective trees
Nmax = 2000                # Number of sampled points
statespace_lo = [-5,-5]
statespace_hi = [5,5]

t = time.time()
sample = []
for i in range(1,Nmax + 1): # Collect sample points each node id is the same as its index in the list
    location = np.random.uniform(statespace_lo, statespace_hi)
    if not isBetween(MAZE,location, 0.15): # If the sample falls in an obstalce within the specified margin, do NOT create a node
        new_node = node(location)
        sample.append(new_node)

# 2) Build Tree for pursuer, set pursuer goal to the region that's diagonally
#    opposite to its starting point to ensure that the FMT tree covers the 
#    whole state space
pursuer = GeometricFMT(statespace_lo, statespace_hi, [-4,4], None, MAZE)
pursuer.solve(1.0, copy.deepcopy(sample), Nmax, 1.5, True)

# Compare the costs of the same node. Best is to do it by location. Since we 
# have the sample, then we loop through the location of the nodes. How do you 
# access the node.loc in both lists using the location? Here's the idea:
#    i) Loop through the locations in sample
#   ii) Have two variables to store the indeces in V_p and V_e 
#  iii) Compare the distances of the the node using the indeces to look it up
#       in each list
#for x in sample:
#    idx_p = [tmp.getid() for tmp in V_p if np.array_equal(tmp.loc, x.loc)]
#    print idx_p
#    idx_p =
#    if x.loc == value:
#        print "i found it!"
#        break

    
# 3) Build Tree for evader. Need to modify the solve method to filter and 
#    remove branches to nodes where the pursuer's cost < evader's cost
#    
evader = GeometricFMT(statespace_lo, statespace_hi, [-4,-4], [4,4], MAZE)
evader.solve(1.0, copy.deepcopy(sample), Nmax)
node_remove = []

for i in range(1,len(evader.V)):
    if (pursuer.V[i].T <= evader.V[i].T and pursuer.V[i].T != None) or evader.V[i].T == None:
        node_remove.append(evader.V[i])
        
for node in node_remove:
    if node in evader.V:
        evader.remove_branch(node)        
evader.plot_all()
t2 = time.time() - t
print "cost = ", evader.cost
print "time of PE-FMT* =", t2
plt.show()
