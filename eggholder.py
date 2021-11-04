import numpy as np
from particle_swarm import ParticleSwarm
import matplotlib.pyplot as plt
import math
import pandas as pd

class EGGHOLDER:

    def __init__(self, is_debug):
        self.is_debug = is_debug
    
    def generate_init_positions(self, seed, swarm_size):
        """
        Generates the initial particle position within function limits.
        """
        rng = np.random.default_rng(seed)
        return [np.ndarray.tolist(rng.uniform(-512,512,2)) for _ in range(swarm_size)]

    def eval_func(self, particle):
        """
        Evaluates the current state by
        calculating the function result.
        """
        x1 = particle[0]
        x2 = particle[1]
        f = -(x2+47)*math.sin(math.sqrt(abs(x2 + x1/2 +47))) -x1*math.sin(math.sqrt(abs(x1-(x2+47))))
        return f
    
    def solve_eggholder(self):
        """
        Calls the PS for the eggholder function problem and plots the results.
        """

        # fig = plt.figure(figsize=(10, 5))
        PS = ParticleSwarm(is_debug=self.is_debug)
        data = []

        headers = ['iterations', 'swarm_size', 776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]
        seeds = [123556]
        # seeds = [776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]
        # iterations_list=[50,100,200,300]
        iterations_list=[300]
        phi_list = [2.0, 2.0]
        # t_size_list=[9, 10, 11]
        # swarm_size_list=[100,200,400,800]
        swarm_size_list=[800]

        for iterations in iterations_list:
            for idx, swarm_size in enumerate(swarm_size_list):
                # plt.subplot(len(swarm_size_list)//2,3,idx+1)
                # plt.title('swarm_size = {:}, iterations = {:}'.format(swarm_size, iterations))
                best_per_seed = []
                for seed in seeds:
                    best_list, _ = PS.pso(init_pos_f = self.generate_init_positions, swarm_size=swarm_size, iterations=iterations, eval_f=self.eval_func, 
                                        seed=seed, phi1=phi_list[0], phi2=phi_list[1],
                                        neighborhood_topology="star", model="full", inertia=False, constriction=False, boundaries = [-512, 512])

                    # plt.plot(range(len(best_list)), best_list, label=str(seed))
                    # plt.legend()

                    print("Best solution: ", best_list[-1])
                    best_per_seed.append(best_list[-1])

                tmp = [iterations, swarm_size]
                tmp.extend(best_per_seed)
                data.append(tmp)
            # plt.show()
        df= pd.DataFrame(data=data, columns= list(map(str, headers)))
        print(df)
        # df.to_excel("../ps_300it_800s_10seeds_ring.xlsx")
       
        # plt.show()


if __name__ == "__main__":
    EGGHOLDER = EGGHOLDER(is_debug=False)
    
    # -959.6407
    print("Global optimum is: ", EGGHOLDER.eval_func([512, 404.2319]))

    EGGHOLDER.solve_eggholder()