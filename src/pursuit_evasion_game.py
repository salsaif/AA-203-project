#pursuit_evasion.py

import evader
import pursuer
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection
import time
np.random.seed(1)

class PursuitEvasionGame(object):
    """docstring for ClassName"""
    def __init__(self, evader, pursuer): 
        self.e = evader
        self.p = pursuer

    def near_capture(self, tree, x,eps):
        n = len(tree)
        dist = min(np.sqrt(100*np.log(n)/(np.pi*n)),eps)
        retval = []
        for i in range(len(tree)):
            if np.linalg.norm(np.array(x.loc)-np.array(tree[i].loc)) <= dist:
                retval.append(tree[i])
        return retval

    def play_game(self, max_iters, eps = 3.0):

        

        x_init_e = evader.node(self.e.x_init)
        x_init_e.setP(None)
        x_init_e.setT(0)
        self.e.V.append(x_init_e)

        x_init_p = pursuer.node(self.p.x_init)
        x_init_p.setP(None)
        x_init_p.setT(0)
        self.p.V.append(x_init_p)

        i = 0
        while i <= max_iters:
        # goal = False
        # while not goal and i <= max_iters:
            i = i + 1
            x_rand_e = np.random.uniform(self.e.ss_lo, self.e.ss_hi)
            x_new_e = self.e.extend(x_rand_e, eps)
            if (x_new_e != None):
                Z_p_near = self.near_capture(self.p.V, x_new_e,eps)
                for z_p_near in Z_p_near:
                    if z_p_near.T < x_new_e.T and x_new_e in self.e.V:
                        print("top")
                        self.e.remove_branch1(x_new_e)
                # if self.e.is_safe(x_new_e):
                #     goal = True
                #     break
            x_rand_p = np.random.uniform(self.p.ss_lo, self.p.ss_hi)
            x_new_p = self.p.extend(x_rand_p, eps)
            if (x_new_p != None):
                Z_e_near = self.near_capture(self.e.V, x_new_p,eps)
                for z_e_near in Z_e_near:
                    if x_new_p.T < z_e_near.T and z_e_near in self.e.V:
                        print("bottom")
                        self.e.remove_branch1(z_e_near)

        plt.figure()
        self.e.plot_everything()
        self.p.plot_everything()

############################ TESTING ################################

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

NOMAZE = np.array([])
# e/p = Evader/PursuerRRT(ss_lo, ss_hi, x_init, "x_goal", obstacles)
#e = evader.EvaderRRT([-5,-5], [5,5], [-4,-4], [4,-4.5], NOMAZE)
#p = pursuer.PursuerRRT([-5,-5], [5,5], [4,4], NOMAZE)
t = time.time()
e = evader.EvaderRRT([-5,-5], [5,5], [-4,-4], [3,0], MAZE)
p = pursuer.PursuerRRT([-5,-5], [5,5], [-4,0], MAZE)
peg = PursuitEvasionGame(e,p)
peg.play_game(500,3.0)
elapsed = time.time() - t
print(elapsed)
plt.show()

