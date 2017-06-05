#pursuit_evasion.py

import evader
import pursuer
import numpy as np
import matplotlib.pyplot as plt
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

            i = i + 1
            x_rand_e = np.random.uniform(self.e.ss_lo, self.e.ss_hi)
            x_new_e = self.e.extend(x_rand_e, eps)
            if (x_new_e != None):
                Z_p_near = self.near_capture(self.p.V, x_new_e,eps)
                for z_p_near in Z_p_near:
                    if z_p_near.T < x_new_e.T and x_new_e in self.e.V:
                        self.e.remove_branch(x_new_e)

            x_rand_p = np.random.uniform(self.p.ss_lo, self.p.ss_hi)
            x_new_p = self.p.extend(x_rand_p, eps)
            if (x_new_p != None):
                Z_e_near = self.near_capture(self.e.V, x_new_p,eps)
                for z_e_near in Z_e_near:
                    if x_new_p.T < z_e_near.T and z_e_near in self.e.V:
                        self.e.remove_branch(z_e_near)


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

t = time.time()
e = evader.EvaderRRT([-5,-5], [5,5], [-4,-4], [3,0], MAZE)
p = pursuer.PursuerRRT([-5,-5], [5,5], [-4,0], MAZE,2.0)
peg = PursuitEvasionGame(e,p)
peg.play_game(500,3.0)
elapsed = time.time() - t
print "run time = ", elapsed
plt.figure()
peg.e.plot_everything()
#peg.p.plot_everything()
print "cost = ", peg.e.cost

plt.show()

