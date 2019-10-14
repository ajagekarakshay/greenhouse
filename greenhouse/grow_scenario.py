import numpy as np
import math
import random


class GrowScenario:
    """Greenhouse scenario:

    tops: height of the top of the plant (0, n]
    bulbs: turned on 1, off 0, broken -1
    bulbs_height: height of the bulbs (should be higher than presence in any given square), defaults to 1.

    """

    def __init__(self, width, height, broken_bulbs=None):
        self.height = height
        self.width = width
        self.tops = np.zeros((height, width))
        self.bulbs = np.zeros((height, width))
        self.bulbs_height = np.ones((height, width)) + 1.0
        self.lux = np.zeros((height, width))
        self.ambient_lux = 1000.0  # Weather
        self.mean = 0.0
        self.std = 0.0

        # Average cost of 100kwh was 20,4 euros in Europe 2017 (https://www.energiauutiset.fi/uutiset/suomen-sahko-yha-edullista.html)
        # One bulb running 24/7 full year consumes 24 * 365 * 400 = 3504000 watt/hrs = 3504 kwh
        # 3504 kwh costs (minimum) around 700 euros (3504 * 0.204).
        # So reducing bulb usage even by a couple of bulbs at any time will result in significant savings.
        # (This does not include savings from replacing the bulbs, etc.)
        self.bulb_watts = 400
        self.bulb_lumens = 10000  # Stetson, probably around 35000-50000

        if broken_bulbs is not None:
            for b in broken_bulbs:
                self.bulbs[b] = -1

    @property
    def presence(self):
        return self.tops

    @presence.setter
    def presence(self, new_presence):
        self.tops = new_presence

    def reinitialize_tops(self, min_height, max_top):
        """Reinitialize plant height with given parameters
        """
        self.tops = min_height + np.random.random((self.height, self.width)) * max_top

    def reinitialize(self, tops=None, ambient_lux=None, std=None, mean=None):
        """Reinitialize plant heights and ambient lux.
        """
        if tops is None:
            means = [0.35, 0.5, 0.65]
            self.std = (random.random() * 0.2) + 0.1 if std is None else std
            self.mean = random.choice(means) if mean is None else mean
            next_ctx = np.zeros((self.height, self.width)) + self.mean + np.random.normal(0, self.std, (self.height, self.width))
            next_ctx[next_ctx < 0] = 0.0
            next_ctx[next_ctx > 1] = 1.0
            self.tops = next_ctx
        else:
            self.tops = tops
            if std is not None:
                self.std = std
            if mean is not None:
                self.mean = mean
        if ambient_lux is None:
            lux_choices = [0.0, 500.0, 1000.0, 3000.0, 5000.0, 7000.0, 9000.0]
            self.ambient_lux = random.choice(lux_choices)
        else:
            self.ambient_lux = ambient_lux

    def reinitialize_broken_bulbs(self, broken_bulbs):
        """Change broken bulbs setting.
        """
        self.bulbs = np.zeros((self.height, self.width))

        for b in broken_bulbs:
            self.bulbs[b] = -1

    def change_bulbs_height(self, new_heights):
        assert np.all(new_heights > self.tops)
        self.bulbs_height = new_heights

    def compute_cost(self, bulbs):
        return self.bulb_watts * np.count_nonzero(bulbs)

    def compute_lux(self, tops=None, bulbs=None, bulbs_height=None):
        """Compute how much "lux" top of the plant in each square in the scenario gets.
        """
        if tops is None:
            tops = self.tops
        if bulbs is None:
            bulbs = self.bulbs
        if bulbs_height is None:
            bulbs_height = self.bulbs_height

        # Do "convolution" trick to speed up the computation by couple of orders of magnitude
        lux = np.zeros((self.height + 4, self.width + 4)) + self.ambient_lux
        tops_copy = np.zeros((self.height + 4, self.width + 4))
        tops_copy[2:self.height+2, 2:self.width+2] = tops

        # All bulbs near the square in question affect lux it gets so we "convolve" the tops_copy with bulb_height
        # and compute based on them into lux.
        for i in range(-2, 3):
            for j in range(-2, 3):
                xf = 2 + i
                yf = 2 + j
                square_distance_factor = 0.2 ** max(abs(i), abs(j))  # Magic number
                bulb_distances = bulbs_height - tops_copy[xf:self.height+xf, yf:self.width+yf]
                lux[xf:self.height+xf, yf:self.width+yf] += (self.bulb_lumens / (bulb_distances ** 2)) * square_distance_factor * bulbs

        return lux[2:self.height+2, 2:self.width+2]












