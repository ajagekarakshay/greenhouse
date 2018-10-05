
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
from home import Home

class AdaptiveHome(Home):

    def __init__(self, w, h):
        super().__init__(w,h)
        self.scenario = Scenario()
        self.presence, self.bulbs = self.scenario.diagonal(self.width, self.height)
        #self.presence, self.bulbs = self.scenario.corners(self.width, self.height)
        #self.presence, self.bulbs = self.scenario.stripes(self.width, self.height)


        self.fig = plt.figure(figsize=(1, 3))

        self.fig.add_subplot(131)
        plt.imshow(self.presence, cmap='gray', interpolation='nearest', vmin=0, vmax=1)

        self.fig.add_subplot(132)
        plt.imshow(self.bulbs, cmap='gray', interpolation='nearest', vmin=-1, vmax=0)

        self.fig.add_subplot(133)
        self.im = plt.imshow(self.luminosity, cmap='gray', interpolation='bilinear', animated=True, vmin=0, vmax=1)


    def updatefig(self, *args):

        #self.presence = self.scenario.random(self.width, self.height)

        for y in range(self.height):
            for x in range(self.width):
                if self.presence[x,y] > 0:
                    if self.bulbs[x,y] > -1:
                        self.luminosity[x,y] = 1 #random.choice([1,2])
                    else:
                        x_near, y_near = self.strategy_find_near_bulb(x,y)
                        if x_near > -1 and y_near > -1:
                            self.luminosity[x_near,y_near] = 1 #random.choice([1,2])
                            #print('bulb found near')
                else:
                    self.luminosity[x,y] = 0

        self.im.set_data(self.luminosity)
        #im.set_cmap("gray")
        #im.update()
        #print("update called")
        return self.im,

    def strategy_find_near_bulb(self, x, y):
        block = ((x-1, y-1), (x, y-1), (x+1,y-1), (x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1), (x-1, y)) # starts from left top
        #print('looking for nearby lamp')
        candidate_bulbs = []
        point = [-1, -1]

        for point in block:
            if point[0] in range(self.width) and point[1] in range(self.height):
                if self.bulbs[point[0], point[1]] > -1:
                    candidate_bulbs.append(point)

        if len(candidate_bulbs):
            point = random.choice(candidate_bulbs)

        return point[0], point[1]

    def run(self):
        #ani = animation.FuncAnimation(self.fig, self.updatefig, interval=50, blit=True)
        self.updatefig()
        plt.show()

