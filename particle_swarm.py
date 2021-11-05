import numpy as np
from random import sample

import os
import matplotlib.pyplot as plt
import imageio
import math
import matplotlib.pyplot as plt

class ParticleSwarm:

    def __init__(self, is_debug):
        self.is_debug = is_debug

    def calculate_constriction(self,phi1, phi2):
        phi = phi1 + phi2
        K = 2 / abs(2 - phi - math.sqrt(phi**2 - 4*phi))
        return K
    
    def calculate_particle_velocity_limits(self, boundaries):
        limits = []
        for boundary in boundaries:
            x_min = np.min(boundary)
            x_max = np.max(boundary)
            limit = (x_max-x_min)/2
            limits.append(limit)
        print(limits)
        return limits
    
    def create_particle_position_gif(self, particles, iterations, seed, model, neighborhood_topology):
        filenames = []
        x_pos = []
        y_pos = []
        for i in range(iterations):
            for particle in particles:
                # if i== iterations-1:
                #     print(particle.historical_x[-1])
                x_pos.append(particle.historical_x[i][0])
                y_pos.append(particle.historical_x[i][1])
            
            # print(x_pos, y_pos)
                # plot the line chart
            plt.title("Iteration: {}".format(i))
            plt.scatter(x_pos, y_pos)
            plt.ylim(-600,600)
            plt.xlim(-600,600)
            plt.xlabel("x1 value")
            plt.ylabel("x2 value")
            
            # create file name and append it to a list
            filename = f'./gifs/{i}.png'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()
            x_pos = []
            y_pos = []
        # build gif
        with imageio.get_writer(f'./gifs/{seed}_{model}_{neighborhood_topology}_vrestriction.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Remove files
        for filename in set(filenames):
            os.remove(filename)

    def pso(self, init_pos_f, iterations, eval_f, seed, 
            swarm_size, phi1, phi2, neighborhood_topology = "star", model="full", 
            inertia=True, constriction=True, boundaries=[[-512,512], [-512,512]], velocity_restriction=True, generate_gifs=False):

            """Particle Swarm Optimization algorithm

            Parameters:
                iterations : maximum number of iterations
                eval_f : evaluation function
                seed: Random seed for particle initialization
                swarm_size: Number of particles in the swarm
                phi1: the φ1 value
                phi2: the φ2 value
                neighborhood_topology: "star" or "ring"
                model: "full", "cognition", "social", or "selfless"
                inertia: Whether to apply intertia to the velocity vector or not.
                constriction: Whether to apply constriction to the velocity vector or not.
                boundaries: list of lists with function boundaries for each dimension.
                velocity_restriction: Whether or not to restrict the max velocity of the particles.

            Returns:
                best_list : list of historical best solutions through the iterations
            """
            
            print("Running the particle swarm optimization algorithm...")

            # initialization
            swarm_list = []
            best_list = []
            ω = 1
            K = 1

            if constriction:
                K = self.calculate_constriction(phi1, phi2)
            
            if velocity_restriction:
                particle_velocity_limits = self.calculate_particle_velocity_limits(boundaries)
               
            # Generate the particles
            initial_positions = init_pos_f(seed, swarm_size)
            for i in range(swarm_size):
                swarm_list.append(Particle(self.is_debug, initial_positions[i], eval_f, phi1, phi2, model, velocity_restriction))
            
            if neighborhood_topology == "ring":
                for particle in swarm_list:
                    particle.set_neighbors(sample(swarm_list, 2))
            
            if self.is_debug:
                print("Swarm size: ", len(swarm_list))
                print("PSO will run for {} iterations.".format(iterations))
                print("Swarm details:") 
                for idx, particle in enumerate(swarm_list):
                    print("Particle number: {}\n Particle x: {}\n Particle v: {}\n"
                    .format(idx, particle.x_vector, particle.v_vector))

            for _ in range(iterations):
            
                for particle in swarm_list:
                    particle.evaluate()

                sorted_particles = sorted(swarm_list, key=lambda particle: particle.p_fitness)
                best_particle = sorted_particles[0]
                second_best_particle = sorted_particles[1]
                if self.is_debug:
                    print("Best particle X vector: ", best_particle.x_vector)
                    print("Best particle X fitness: ", best_particle.x_fitness)
                    print("Best particle P vector: ", best_particle.p_vector)
                    print("Best particle P fitness: ", best_particle.p_fitness)
                best_list.append(best_particle.p_fitness)

                for particle in swarm_list:
                    particle.adjust_v(best_particle, ω, K, second_best_particle, particle_velocity_limits)
                    particle.move()
                
                if inertia:
                    ω*=0.995
                
            random_particles = sample(swarm_list,2)
            print("Final interia: ", ω)
    
            if generate_gifs:
                self.create_particle_position_gif(swarm_list, iterations, seed, model, neighborhood_topology)

            return best_list, random_particles

class Particle:

    def __init__(self, is_debug, position, eval_f, phi1, phi2, model, velocity_restriction):
        self.is_debug = is_debug
        self.model = model
        self.velocity_restriction = velocity_restriction
        self.historical_v = []
        self.historical_x = []
        self.eval_f = eval_f
        self.x_vector = position # Initial position
        self.p_vector = self.x_vector
        self.v_vector = self.initialize_v_vector(len(position)) # Gradient
        self.historical_v.append(self.v_vector)
        self.x_fitness = 1e50
        self.p_fitness = 1e50
        self.neighbors = None

        if model == "social" or model == "selfless":
            self.phi1 = 0
            self.phi2 = phi2
        elif model == "cognition":
            self.phi1 = phi1
            self.phi2 = 0
        else:
            self.phi1 = phi1
            self.phi2 = phi2
    
    def move(self):

        for i in range(len(self.x_vector)):
            tmp = self.x_vector[i] + self.v_vector[i]
            if -512 <= tmp <= 512:
                self.x_vector[i] = tmp

        
        
    def evaluate(self):
        self.x_fitness = self.eval_f(self.x_vector)
        if self.x_fitness < self.p_fitness:
            self.p_fitness = self.x_fitness
            self.p_vector = self.x_vector.copy()
            
        # print(self.p_fitness)
        self.historical_x.append(self.x_vector.copy())

    def adjust_v(self, best_particle, ω, K, second_best_particle, limits):

        if self.neighbors:
            best_neighbor = sorted(self.neighbors, key=lambda particle: particle.p_fitness)[0]
            self.v_vector = [self.v_vector[i] + 
                            self.phi1*np.random.uniform(0,1)*(self.p_vector[i] - self.x_vector[i]) + 
                            self.phi2*np.random.uniform(0,1)*(best_neighbor.p_vector[i] - self.x_vector[i])
                            for i in range(len(self.x_vector))]
        else:
            if self.model != "selfless" :
                self.v_vector = [K*(ω*self.v_vector[i] + 
                                self.phi1*np.random.uniform(0,1)*(self.p_vector[i] - self.x_vector[i]) + 
                                self.phi2*np.random.uniform(0,1)*(best_particle.p_vector[i] - self.x_vector[i]))
                                for i in range(len(self.x_vector))]
            else:
                if best_particle != self:
                    self.v_vector = [self.v_vector[i] + 
                                    self.phi1*np.random.uniform(0,1)*(self.p_vector[i] - self.x_vector[i]) + 
                                    self.phi2*np.random.uniform(0,1)*(best_particle.p_vector[i] - self.x_vector[i])
                                    for i in range(len(self.x_vector))]
                else:
                    self.v_vector = [self.v_vector[i] + 
                                    self.phi1*np.random.uniform(0,1)*(self.p_vector[i] - self.x_vector[i]) + 
                                    self.phi2*np.random.uniform(0,1)*(second_best_particle.p_vector[i] - self.x_vector[i])
                                    for i in range(len(self.x_vector))]
        
        if self.velocity_restriction:
            for i in range(len(self.v_vector)):
                if self.v_vector[i] < -limits[i]:
                    self.v_vector[i] = self.v_vector[i]/limits[i]
                elif self.v_vector[i] > limits[i]:
                    self.v_vector[i] = self.v_vector[i]/limits[i]
        
        self.historical_v.append(self.v_vector)

        
    def initialize_v_vector(self, size):
        return np.ndarray.tolist(np.random.uniform(-5,5,size))
    
    def set_neighbors(self, neighbors):
        self.neighbors = neighbors


def generate_init_positions(seed, swarm_size):
        """
        Generates the initial particle position within function limits.
        """
        rng = np.random.default_rng(seed)
        return [np.ndarray.tolist(rng.uniform(-512,512,2)) for _ in range(swarm_size)]


def eval_func(particle):
        """
        Evaluates the current state by
        calculating the function result.
        """
        # print(particle)

        x1 = particle[0]
        x2 = particle[1]
        f = -(x2+47)*math.sin(math.sqrt(abs(x2 + x1/2 +47))) -x1*math.sin(math.sqrt(abs(x1-(x2+47))))
        return f

if __name__ == "__main__":
    PSO = ParticleSwarm(is_debug=False)
    best_list, random_particles = PSO.pso(init_pos_f=generate_init_positions, iterations=120, 
                                        eval_f=eval_func, seed=776, swarm_size=500, phi1=2.0, phi2=2.0,
                                        neighborhood_topology = "star", model="full", inertia=False, constriction=False,
                                        boundaries=[[-512,512], [-512,512]], velocity_restriction=True)

    fig, ax = plt.subplots(1,2)
    fig.text(0.5, 0.04, 'iterations', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'velocity', va='center', rotation='vertical', fontsize=12)
    for idx, particle in enumerate(random_particles):
        x_v, y_v = map(list, zip(*particle.historical_v))
        ax.flatten()[idx].set_title('Particle {}'.format(idx), fontsize=12)
        ax.flatten()[idx].plot(range(len(x_v)),x_v, label="x velocity")
        ax.flatten()[idx].plot(range(len(y_v)),y_v, label="y velocity")
        # ax.flatten()[idx].set_yscale('log')
        ax.flatten()[idx].legend()

    plt.show()
    
def constriction_experimentation():
    PSO = ParticleSwarm(is_debug=False)
    phi_list = [(2,2), (2.2,2.2), (2.5,2.5), (3,3), (4,4), (5,5), (6,6)]
    constriction_list = []
    phi_total_list = []
    for phis in phi_list:
        phi_total_list.append(phis[0] + phis[1])
        constriction_list.append(PSO.calculate_constriction(phis[0],phis[1]))

    plt.ylabel("constriction value")
    plt.xlabel("φ value")
    plt.plot(phi_total_list, constriction_list)
    plt.show()