import numpy as np
import copy
from chromosome import UAVFleetChromosome

class GeneticStrategyEngine:
    """
    Multi-Objective Genetic Algorithm Engine with:
    - Composite weighted fitness evaluation + hard constraint penalties
    - Tournament Selection & Elitism
    - Blended Geometric Crossover (BLX-alpha) + Discrete Uniform Sensor Crossover
    - Adaptive Gaussian Spatial Mutation + Bit-Flip Sensor Mutation
    """
    def __init__(self, env, pop_size=100, mutation_rate=0.10, tournament_k=4, elitism_count=4):
        self.env = env
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elitism_count = elitism_count
        self.population = [UAVFleetChromosome(env=env) for _ in range(pop_size)]

    def evaluate_fitness(self, chrom):
        covered_mask = np.zeros((self.env.grid_size, self.env.grid_size), dtype=bool)
        total_energy = 0.0
        max_flight_time = 0.0
        hazard_exposure = 0.0
        nfz_violations = 0
        range_penalty = 0.0
        max_battery = 2400.0

        sensor_radii = {0: 0, 1: 3, 2: 5}
        sensor_power = {0: 0.0, 1: 4.5, 2: 11.0}

        for u in range(chrom.num_uavs):
            prev_pt = self.env.base_depot
            uav_energy = 0.0
            uav_time = 0.0

            for w in range(chrom.num_waypoints):
                x, y, v, s = chrom.genes[u, w]
                curr_pt = np.array([x, y])
                dist = np.linalg.norm(curr_pt - prev_pt)
                dt = dist / max(v, 1.0)

                # Propulsion + active sensor energy model
                step_energy = (0.08 * (v ** 2) * dist) + (sensor_power[int(s)] * dt)
                uav_energy += step_energy
                uav_time += dt

                # Check line-of-sight intersections with NFZ
                for t in np.linspace(0, 1, 8):
                    ix, iy = prev_pt + t * (curr_pt - prev_pt)
                    if self.env.is_in_nofly_zone(ix, iy):
                        nfz_violations += 1
                    hx = int(np.clip(round(ix), 0, self.env.grid_size - 1))
                    hy = int(np.clip(round(iy), 0, self.env.grid_size - 1))
                    hazard_exposure += self.env.hazard_grid[hy, hx] * 0.25

                # Mark coverage footprint around waypoint
                rad = sensor_radii[int(s)]
                if rad > 0:
                    cx, cy = int(round(x)), int(round(y))
                    y_min, y_max = max(0, cy - rad), min(self.env.grid_size, cy + rad + 1)
                    x_min, x_max = max(0, cx - rad), min(self.env.grid_size, cx + rad + 1)
                    covered_mask[y_min:y_max, x_min:x_max] = True

                prev_pt = curr_pt

            total_energy += uav_energy
            max_flight_time = max(max_flight_time, uav_time)
            if uav_energy > max_battery:
                range_penalty += (uav_energy - max_battery) * 2.5

        coverage_score = np.sum(self.env.priority_grid[covered_mask])
        fitness = (1.6 * coverage_score) - (0.04 * total_energy) - (1.2 * max_flight_time) - (1.8 * hazard_exposure) - (80.0 * nfz_violations) - range_penalty
        chrom.fitness = fitness
        return fitness

    def tournament_selection(self):
        contenders = np.random.choice(self.population, size=self.tournament_k, replace=False)
        return max(contenders, key=lambda c: c.fitness)

    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        alpha = np.random.uniform(-0.15, 1.15, size=parent1.genes[:, :, :3].shape)
        child1.genes[:, :, :3] = alpha * parent1.genes[:, :, :3] + (1 - alpha) * parent2.genes[:, :, :3]
        child2.genes[:, :, :3] = (1 - alpha) * parent1.genes[:, :, :3] + alpha * parent2.genes[:, :, :3]

        # Discrete uniform crossover for sensor schedule
        mask = np.random.rand(parent1.num_uavs, parent1.num_waypoints) > 0.5
        child1.genes[:, :, 3] = np.where(mask, parent1.genes[:, :, 3], parent2.genes[:, :, 3])
        child2.genes[:, :, 3] = np.where(mask, parent2.genes[:, :, 3], parent1.genes[:, :, 3])

        child1.repair_constraints()
        child2.repair_constraints()
        return child1, child2

    def mutate(self, chrom, generation, max_generations):
        sigma = 4.5 * (1.0 - (generation / max_generations) * 0.75)
        for u in range(chrom.num_uavs):
            for w in range(chrom.num_waypoints):
                if np.random.rand() < self.mutation_rate:
                    chrom.genes[u, w, 0] += np.random.normal(0, sigma)
                    chrom.genes[u, w, 1] += np.random.normal(0, sigma)
                    chrom.genes[u, w, 2] += np.random.normal(0, 1.5)
                    if np.random.rand() < 0.35:
                        chrom.genes[u, w, 3] = np.random.choice([0, 1, 2])
        chrom.repair_constraints()

    def step_generation(self, gen, max_gen):
        for chrom in self.population:
            if chrom.fitness == -np.inf:
                self.evaluate_fitness(chrom)
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        new_pop = [copy.deepcopy(self.population[i]) for i in range(self.elitism_count)]

        while len(new_pop) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = self.crossover(p1, p2)
            self.mutate(c1, gen, max_gen)
            self.mutate(c2, gen, max_gen)
            self.evaluate_fitness(c1)
            self.evaluate_fitness(c2)
            new_pop.extend([c1, c2])

        self.population = new_pop[:self.pop_size]
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        best_f = self.population[0].fitness
        avg_f = np.mean([c.fitness for c in self.population])
        return best_f, avg_f
