import numpy as np
from particle_swarm import ParticleSwarm
import matplotlib.pyplot as plt
import math
import pandas as pd

class SCHWEFEL:

    def __init__(self, is_debug):
        self.is_debug = is_debug
    
    def generate_init_positions(self, seed, swarm_size):
        """
        Generates the initial particle position within function limits.
        """
        rng = np.random.default_rng(seed)
        return [np.ndarray.tolist(rng.uniform(-512,512,4)) for _ in range(swarm_size)]

    def eval_func(self, particle):
        """
        Evaluates the current state by
        calculating the function result.
        """

        # print (np.sin(np.sqrt(np.abs(particle))))
        f = np.sum(np.multiply(np.negative(particle), np.sin(np.sqrt(np.abs(particle))))) + 418.982887*4
        return f
    
    def solve_schwefel(self):
        """
        Calls the PS for the schwefel function and plots the results.
        """

        fig = plt.figure(figsize=(10, 5))
        PS = ParticleSwarm(is_debug=self.is_debug)
        data = []

        headers = ['iterations', 'swarm_size', 776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]
        seeds = [776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]
        # iterations_list=[20,50,100,500]
        iterations_list=[500]
        phi_list = [2.1, 2.1]
        # t_size_list=[9, 10, 11]
        # swarm_size_list=[10,50,100,1000]
        swarm_size_list=[1000]

        for iterations in iterations_list:
            for idx, swarm_size in enumerate(swarm_size_list):
                # plt.subplot(len(swarm_size_list)//2,3,idx+1)
                plt.title('swarm_size = {:}'.format(swarm_size))
                best_per_seed = []
                for seed in seeds:
                    best_list, _ = PS.pso(init_pos_f = self.generate_init_positions, swarm_size=swarm_size, iterations=iterations, eval_f=self.eval_func, 
                                        seed=seed, phi1=phi_list[0], phi2=phi_list[1],
                                        neighborhood_topology="star", model="full", inertia=True, constriction=True)

                    plt.plot(range(len(best_list)), best_list, label=str(seed))
                    plt.legend()

                    print("Best solution: ", best_list[-1])
                    best_per_seed.append(best_list[-1])

                tmp = [iterations, swarm_size]
                tmp.extend(best_per_seed)
                data.append(tmp)
            plt.show()
        df= pd.DataFrame(data=data, columns= list(map(str, headers)))
        print(df)
        # df.to_excel("../ps_500it_variable_s_10seeds_star_inertia_constriction.xlsx")
       
        plt.show()


if __name__ == "__main__":
    SCHWEFEL = SCHWEFEL(is_debug=False)
    
    # 0
    print("Global optimum is: ", SCHWEFEL.eval_func([420.968746,420.968746,420.968746,420.968746]))

    SCHWEFEL.solve_schwefel()