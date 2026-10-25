import numpy as np

class UAVFleetChromosome:
    """
    Hybrid Multi-Segment Chromosome encoding deployment trajectories, speed profiles,
    and sensor activation schedules for a fleet of K UAVs across M waypoints each.
    Each gene tuple: (x, y, v, s)
      - x, y: Continuous spatial coordinates in [0, grid_size-1]
      - v: Flight velocity in [5.0, 20.0] m/s
      - s: Sensor power mode in {0: Off, 1: Low-Power, 2: High-Resolution}
    """
    def __init__(self, num_uavs=3, num_waypoints=7, env=None):
        self.num_uavs = num_uavs
        self.num_waypoints = num_waypoints
        self.env = env
        self.grid_size = env.grid_size if env else 50
        self.genes = np.zeros((num_uavs, num_waypoints, 4))
        self.fitness = -np.inf
        self.randomize()

    def randomize(self):
        for u in range(self.num_uavs):
            for w in range(self.num_waypoints):
                self.genes[u, w, 0] = np.random.uniform(2, self.grid_size - 2)
                self.genes[u, w, 1] = np.random.uniform(2, self.grid_size - 2)
                self.genes[u, w, 2] = np.random.uniform(5.0, 20.0)
                self.genes[u, w, 3] = np.random.choice([0, 1, 2], p=[0.15, 0.45, 0.40])
        self.repair_constraints()

    def repair_constraints(self):
        """Dynamic geometric repair pushing waypoints outside hard No-Fly Zones and grid bounds."""
        self.genes[:, :, 0] = np.clip(self.genes[:, :, 0], 1.0, self.grid_size - 2.0)
        self.genes[:, :, 1] = np.clip(self.genes[:, :, 1], 1.0, self.grid_size - 2.0)
        self.genes[:, :, 2] = np.clip(self.genes[:, :, 2], 5.0, 20.0)
        self.genes[:, :, 3] = np.round(np.clip(self.genes[:, :, 3], 0, 2))

        if self.env is not None:
            for u in range(self.num_uavs):
                for w in range(self.num_waypoints):
                    x, y = self.genes[u, w, 0], self.genes[u, w, 1]
                    for (rx, ry, rw, rh) in self.env.nofly_rects:
                        if rx <= x <= rx + rw and ry <= y <= ry + rh:
                            # Project waypoint to nearest boundary edge of the NFZ rectangle
                            dists = [abs(x - rx), abs(x - (rx + rw)), abs(y - ry), abs(y - (ry + rh))]
                            min_idx = np.argmin(dists)
                            if min_idx == 0: self.genes[u, w, 0] = rx - 1.0
                            elif min_idx == 1: self.genes[u, w, 0] = rx + rw + 1.0
                            elif min_idx == 2: self.genes[u, w, 1] = ry - 1.0
                            else: self.genes[u, w, 1] = ry + rh + 1.0
