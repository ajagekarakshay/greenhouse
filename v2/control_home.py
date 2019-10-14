
# Scenarios:
#   use windows, use doors
#   failed bulb- use others
#   intensity vs cost- turn few bulb on with higher intensity instead of turning all on- saving cost by using one high intensity bulb

# MAPE will have a GA with f.f Cost, Keep it lit for occupants
#      GA: mutations turn ON, Turn off
#           ON= costs 10 units per step, OFF= zero units

# Awareness
#   Goal: Add to f.f. reduce polution, some bulb type create more polution than others
#   Context: roof and wall window control system awareness, linked to resource awareness
#       Resoruce: Amount and location of Wall and roof top windows
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
from scenario import Scenario
from v2.home import Home

class ControlHome(Home):

    def __init__(self, w, h):
        super().__init__(w,h)
        self.scenario = Scenario()
        self.width_bound = [1, w-2]
        self.height_bound = [1, h-2]
        self.presence, self.bulbs = self.scenario.corners2(self.height, self.width)

        self.init_figures()
        self.steps = 0
        self.MAX_STEPS = 1

        self.react()

    def react(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.presence[y,x] > 0:
                    if self.bulbs[y,x] > -1:
                        self.luminosity[y,x] = 1 #random.choice([1,2])
                else:
                    self.luminosity[y,x] = 0

    # ----------  Simulation and visualization elements ----------

    def updatefig(self, *args):

        print('------STEP-----',self.steps)
        self.steps += 1
        if self.steps > self.MAX_STEPS:
            self.ani.event_source.stop() # it takes a while for the animation loop to notice the event
        else:
            self.im.set_data(self.luminosity)
            self.luminosity_im.set_data(self.luminosity_extrapolate(np.copy(self.luminosity)))
        plt.grid()
        plt.grid()
        return self.im, self.luminosity_im

    def luminosity_extrapolate(self, luminosity):
        #there must some function doing this interpolation?
        for y in range(self.height_bound[0], self.height_bound[1]):
            for x in range(self.width_bound[0], self.width_bound[1]):
                if self.bulbs[y,x] > -1:
                    if luminosity[y,x] > 0:
                        block = ((y-1, x-1), (y, x-1), (y+1,x-1), (y+1, x), (y+1, x+1), (y, x+1), (y-1, x+1), (y-1, x))
                        for point in block:
                            if point[1] in range(self.width_bound[0], self.width_bound[1]) and point[0] in range(self.height_bound[0], self.height_bound[1]):
                                luminosity[point[0],point[1]] += 0.15*luminosity[y,x]
                else:
                    luminosity[y,x] = 0

        return luminosity

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=50, blit=True)
        plt.show()

