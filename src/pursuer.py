# pursuer.py

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection
np.random.seed(2)

class node(object):
    _id = 0
    def __init__(self, loc):
        self.loc = loc
        self._id = node._id
        node._id += 1
        self.parent = None
        self.children = []
        self.T = None
        
    def setP(self, parent):
        self.parent = parent

    def addChild (self, child):
        self.children.append(child)
        
    def setT(self, T):
        self.T = T
        
    def getid(self):
        return self._id
        
    def __repr__(self):
        return "%s"%(self.loc)
        
class PursuerRRT(object):
    def __init__(self, ss_lo, ss_hi, x_init, obstacles, sp):
        self.ss_lo = np.array(ss_lo)
        self.ss_hi = np.array(ss_hi)
        self.x_init = np.array(x_init)
        self.obstacles = obstacles
        self.n = 1
        self.V = []
        self.sp = sp

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
            x_new.setT(x_near.T + self.sp*np.linalg.norm(np.array(x_new.loc)-np.array(x_near.loc)))
            Z_near = self.near(x_new,eps)
            if Z_near:
                cmin, zmin = self.best_wire(Z_near, x_new, x_near, eps)
                        
            x_new.setT(zmin.T + self.sp*np.linalg.norm(np.array(x_new.loc)-np.array(zmin.loc)))
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
            dist = self.sp*np.linalg.norm(np.array(z_new)-np.array(Z_near[i].loc))
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
                dist = self.sp*np.linalg.norm(np.array(x_new.loc)-np.array(Z_near[i].loc))
                T_near = Z_near[i].T
                if self.is_free_motion(self.obstacles, x_new.loc, Z_near[i].loc) and x_new.T+dist < T_near:
                    Z_near[i].setP(x_new)
                    x_new.addChild(Z_near[i])

        
    def near(self, x,eps):
        dist = min(np.sqrt(100*np.log(self.n)/(np.pi*self.n)), eps)
        retval = []
        for i in range(len(self.V)):
            if np.linalg.norm(np.array(x.loc)-np.array(self.V[i].loc)) <= dist:
                retval.append(self.V[i])
        return retval

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
        for i in range(len(self.V)):
            nodes[i,:] = self.V[i].loc

        plt.scatter(nodes[:,0], nodes[:,1],color="blue")
        plt.scatter(self.x_init[0], self.x_init[1], color="red", s=30, zorder=10)
        plt.annotate(r"pursuer", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

    def solve(self, eps, max_iters = 1000):
        x_init = node(self.x_init)
        x_init.setP(None)
        x_init.setT(0)
        self.V.append(x_init)

        i = 0
        while i <= max_iters:
            i = i + 1
            x_rand_e = np.random.uniform(self.ss_lo, self.ss_hi)
            self.extend(x_rand_e,eps)

        self.plot_everything()

