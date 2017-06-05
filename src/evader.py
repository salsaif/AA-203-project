# evader.py

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection
np.random.seed(1)

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
        
class EvaderRRT(object):
    def __init__(self, ss_lo, ss_hi, x_init, x_goal, obstacles):
        self.ss_lo = np.array(ss_lo)
        self.ss_hi = np.array(ss_hi)
        self.x_init = np.array(x_init)
        self.x_goal = np.array(x_goal)
        margin = 0.25
        x_goal_hi = x_goal[0]+margin
        x_goal_lo = x_goal[0]-margin
        y_goal_hi = x_goal[1]+margin
        y_goal_lo = x_goal[1]-margin
        self.X_goal = [x_goal_lo, y_goal_lo, x_goal_hi, y_goal_hi]
        self.obstacles = obstacles
        self.n = 1
        self.V = []
        self.cost = None

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def find_nearest(self, x):
        num_states = len(self.V)
        distances = np.zeros(num_states)
        for i in range(num_states):
            distances[i] = np.linalg.norm(np.array(x)-np.array(self.V[i].loc))
        idx = np.argmin(distances)
        return idx

    def steer_towards(self, x, y, eps):
        dist = np.linalg.norm(np.array(x)-np.array(y))
        if dist < eps:
            return y
        return x + eps*(y-x)/dist

    def extend(self,x_rand,eps):
        x_near = self.V[self.find_nearest(x_rand)]
        x_new = node(self.steer_towards(x_near.loc, x_rand, eps))
        if self.is_free_motion(self.obstacles, x_near.loc, x_new.loc):
            self.V.append(x_new)
            x_new.setT(x_near.T + np.linalg.norm(np.array(x_new.loc)-np.array(x_near.loc)))
            Z_near = self.near(x_new, eps)
            if Z_near:
                cmin, zmin = self.best_wire(Z_near, x_new, x_near, eps)
                        
            x_new.setT(zmin.T + np.linalg.norm(np.array(x_new.loc)-np.array(zmin.loc)))
            x_new.setP(zmin)
            zmin.addChild(x_new)
            
            if Z_near:
                self.rewire(zmin, Z_near, x_new, eps)
            
            self.n = self.n+1
            return x_new
        return None

    def best_wire(self, Z_near, x_new, x_near, eps):
        cmin = x_new.T
        zmin = x_near
        for i in range(len(Z_near)):
            z_new = self.steer_towards(Z_near[i].loc,x_new.loc,eps)
            dist = np.linalg.norm(np.array(z_new)-np.array(Z_near[i].loc))
            if self.is_free_motion(self.obstacles, Z_near[i].loc, z_new) and np.all(z_new == x_new.loc) and Z_near[i].T+dist < cmin:
                cmin = Z_near[i].T + dist
                zmin = Z_near[i]
            return cmin,zmin

    def rewire(self, zmin, Z_near, x_new, eps):
        if zmin in Z_near:
            Z_near.remove(zmin)
        for i in range(len(Z_near)):
            z_near = self.steer_towards(x_new.loc,Z_near[i].loc,eps)
            if np.all(z_near == Z_near[i].loc):
                dist = np.linalg.norm(np.array(x_new.loc)-np.array(Z_near[i].loc))
                T_near = Z_near[i].T
                if self.is_free_motion(self.obstacles, x_new.loc, Z_near[i].loc) and x_new.T+dist < T_near:
                    Z_near[i].setP(x_new)
                    x_new.addChild(Z_near[i])

        
    def near(self, x, eps):
        dist = min(np.sqrt(100*np.log(self.n)/(np.pi*self.n)),eps)
        retval = []
        for i in range(len(self.V)):
            if np.linalg.norm(np.array(x.loc)-np.array(self.V[i].loc)) <= dist:
                retval.append(self.V[i])
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

    def is_safe(self, x):
        if x.loc[0] > self.X_goal[0] and x.loc[0] < self.X_goal[2]:
            if x.loc[1] > self.X_goal[1] and x.loc[1] < self.X_goal[3]:
                return True
        return False

    def plot_tree(self, **kwargs):
        plot_line_segments([(self.V[i].parent.loc, self.V[i].loc) for i in range(len(self.V)) if self.V[i].parent != None], **kwargs)

    def plot_path(self, path, **kwargs):
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], **kwargs)

    def plot_solution_path(self, goalnode, **kwargs):
        solution_path_node = [goalnode]
        solution_path = [goalnode.loc]
        while np.all(solution_path[0] != self.x_init):
            parent = solution_path_node[0].parent
            solution_path_node = [parent] + solution_path
            solution_path = [parent.loc] + solution_path
        self.plot_path(solution_path, **kwargs)

    def plot_everything(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        self.plot_tree(color="blue", linewidth=.5, label="RRT tree")
        nodes = np.zeros((self.n,2))
        success = False
        for i in range(len(self.V)):
            nodes[i,:] = self.V[i].loc
            if self.is_safe(self.V[i]):
                goalnode = self.V[i]
                success = True
        if success:
            self.plot_solution_path(goalnode, color="green", linewidth=2, label="solution path")
            self.cost = goalnode.T

        
        plt.scatter(nodes[:,0], nodes[:,1])
        plt.scatter(self.x_init[0], self.x_init[1], color="green", s=30, zorder=10)
        plt.annotate(r"$e_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        if success:
            plt.scatter(goalnode.loc[0], goalnode.loc[1], color="green", s=30, zorder=10)
            plt.annotate(r"$x_{goal}$", goalnode.loc[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        xgoal = np.linspace(self.x_goal[0]-0.5,self.x_goal[0]+0.5,10)
        ygoal1 = (self.x_goal[1]-0.5)*np.ones(10)
        ygoal2 = (self.x_goal[1]+0.5)*np.ones(10)
        plt.fill_between(xgoal,ygoal1,ygoal2,color="green")

    def solve(self, eps, max_iters = 1000):
        x_init = node(self.x_init)
        x_init.setP(None)
        x_init.setT(0)
        self.V.append(x_init)

        success = False
        i = 0
        while not success and i <= max_iters:
            i = i + 1
            x_rand_e = np.random.uniform(self.ss_lo, self.ss_hi)
            x_new_e = self.extend(x_rand_e,eps)
            if (x_new_e != None):
                if self.is_safe(x_new_e):
                    success = True

        self.plot_everything()
