import numpy as np

class HazardGridEnvironment:
    """
    Represents the 50x50 discretized 2D mission space containing:
    - Priority disaster score matrix P(x,y)
    - Environmental hazard intensity matrix H(x,y)
    - Binary hard No-Fly Zone (NFZ) obstacle mask
    """
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.base_depot = np.array([2.0, 2.0])
        self.priority_grid = np.zeros((grid_size, grid_size))
        self.hazard_grid = np.zeros((grid_size, grid_size))
        self.nofly_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self._build_environment()

    def _add_gaussian(self, grid, cx, cy, sigma, amplitude):
        x = np.arange(0, self.grid_size, 1, float)
        y = x[:, np.newaxis]
        grid += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    def _build_environment(self):
        # High-priority disaster zones
        self._add_gaussian(self.priority_grid, 15, 38, 6, 10.0)
        self._add_gaussian(self.priority_grid, 38, 15, 7, 12.0)
        self._add_gaussian(self.priority_grid, 35, 38, 5, 9.0)
        self._add_gaussian(self.priority_grid, 12, 14, 5, 8.0)

        # Environmental hazard zones (wind/radiation pockets)
        self._add_gaussian(self.hazard_grid, 25, 25, 6, 8.0)
        self._add_gaussian(self.hazard_grid, 18, 28, 4, 6.0)
        self._add_gaussian(self.hazard_grid, 32, 22, 5, 7.0)

        # Hard No-Fly Zones (x, y, width, height)
        self.nofly_rects = [(20, 10, 6, 8), (10, 22, 7, 5), (28, 30, 6, 6)]
        for (rx, ry, rw, rh) in self.nofly_rects:
            self.nofly_grid[ry:ry+rh, rx:rx+rw] = True

    def is_in_nofly_zone(self, x, y):
        ix = int(np.clip(round(x), 0, self.grid_size - 1))
        iy = int(np.clip(round(y), 0, self.grid_size - 1))
        return self.nofly_grid[iy, ix]
