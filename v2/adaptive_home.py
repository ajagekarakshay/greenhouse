
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
#import sort
from scenario import Scenario
from v2.control_home import ControlHome

class AdaptiveHome(ControlHome):

    def __init__(self, w, h):
        super().__init__(w,h)
        self.width_bound = [1, w-2]
        self.height_bound = [1, h-2]
        self.scenario = Scenario()
        #self.presence, self.bulbs = self.scenario.diagonal(self.width, self.height)
        #self.presence, self.bulbs = self.scenario.corners(self.width, self.height)
        self.presence, self.bulbs = self.scenario.corners2(self.height, self.width)
        #self.presence, self.bulbs = self.scenario.stripes(self.width, self.height)
        self.plans = []
        self.generate_plans(20)

    #------  Fixed Strategies Generator ----------------

    # This generates a fixed set of plans based on the scenario
    def generate_plans(self, num):
        for i in range(num):
            luminosity_plan = np.copy(self.luminosity)
            for y in range(self.height):
                for x in range(self.width):
                    if self.presence[y,x] > 0:
                        if self.bulbs[y,x] > -1:
                            luminosity_plan [y,x] = 1 #random.choice([1,2])
                        else:
                            y_near, x_near = self.strategy_find_near_bulb(y,x)
                            #if x_near > -1 and y_near > -1:
                            luminosity_plan[y_near,x_near] = 1 #random.choice([1,2])
                                #print('bulb found near')
                    #else:
                    #    self.luminosity[x,y] = 0
            self.plans.append(luminosity_plan)

    def strategy_find_near_bulb(self, y, x):
        block = ((y-1, x-1), (y, x-1), (y+1,x-1), (y+1, x), (y+1, x+1), (y, x+1), (y-1, x+1), (y-1, x))
        candidate_bulbs = []
        point = [-1, -1]

        for point in block:
            if point[0] in range(self.height) and point[1] in range(self.width):
                if self.bulbs[point[0], point[1]] > -1:
                    candidate_bulbs.append(point)

        if len(candidate_bulbs) > 0:
            point = random.choice(candidate_bulbs)

        #print(point)
        return point[0], point[1]

    #---------- MAPE---------------
    def monitor(self):
        # the bulbs, presence and luminsity data simulates monitoring
        self.analyse()

    def analyse(self):
        # due to simplicity, the example does not require analysis stage
        self.plan()

    def plan(self):

        if len(self.plans):
            plans_fitness=[]
            for plan in self.plans:
                fitness = self.fitness(plan)
                print('Fit',fitness)
                plans_fitness.append((fitness, plan))

            plans_sorted = sorted(plans_fitness, key=self.plan_key, reverse=True)

            self.execute (plans_sorted[0])
        else:
            print('ERROR: MAPE has no pre-defined plans to apply.')

    def execute(self, plan):
        self.luminosity = plan[1]
        print('Plan Fitness:', plan[0])

    def plan_key(self, v):
        return v[0]

    # ------------- Evaluation -------
    def fitness(self, plan):#evalInd4
        score = 0
        for y in range(self.height): # top left is (X=0,Y=0)
            for x in range(self.width):
                if plan[y,x] > 0:
                    presence_score = self.presence_in_radius2(1, x, y, plan)
                    if self.bulbs[y,x] > -1 and presence_score > 0:
                        score += presence_score
                    else:
                        score -= 1 #penalty for using a broken bulb
        return (float(score),)

    def presence_in_radius2 (self, radius, x, y, plan):
        block = ((y-1, x-1), (y, x-1), (y+1,x-1), (y+1, x), (y+1, x+1), (y, x+1), (y-1, x+1), (y-1, x)) # starts from left top

        presence_score = 0

        if self.presence[y,x] > 0:
            presence_score += 5

        for point in block:
            if point[0] in range(self.height) and point[1] in range(self.width):
                if self.presence[point[0], point[1]] > 0:
                    presence_score += 1

        return presence_score

    #-------------- Simulation stuff -------------

    def updatefig(self, *args):
        #MAPE loop
        self.monitor()  # the MAPE full chain is called from monitor
        return super().updatefig(*args)



    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=50, blit=True)
        plt.show()

