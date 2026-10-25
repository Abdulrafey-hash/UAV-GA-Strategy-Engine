from environment import HazardGridEnvironment
from ga_engine import GeneticStrategyEngine

def run_simulation(generations=150):
    print("Initializing 50x50 Disaster & Hazard Grid Environment...")
    env = HazardGridEnvironment(grid_size=50)
    engine = GeneticStrategyEngine(env, pop_size=100, mutation_rate=0.10, tournament_k=4, elitism_count=4)

    print(f"Running Multi-Objective Genetic Algorithm for {generations} Generations...")
    for gen in range(1, generations + 1):
        best_f, avg_f = engine.step_generation(gen, generations)
        if gen == 1 or gen % 25 == 0 or gen == generations:
            print(f"Generation {gen:03d}/{generations:03d} | Best Fitness: {best_f:8.2f} | Avg Fitness: {avg_f:8.2f}")

    print("Simulation complete! Optimal UAV fleet schedule evolved successfully.")

if __name__ == "__main__":
    run_simulation()
