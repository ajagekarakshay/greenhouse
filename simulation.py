
# MAPE will have a GA with f.f Cost, Keep it lit for occupants
#      GA: mutations turn ON, Turn off
#           ON= costs 10 units per step, OFF= zero units

# Awareness
#   Goal: Add to f.f. reduce polution, some bulb type create more polution than others
#   Context: roof and wall window control system awareness, linked to resource awareness
#       Resoruce: Amount and location of Wall and roof top windows
#
import numpy as np
from deap import base, creator, algorithms, benchmarks, tools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random

def presence_scenario_a(w, h):
    presence = np.zeros((w, h)) #np.random.randint(2, size=(width,height))
    for i in range(w):
        presence [i,i] = 1
    return presence

#--------------------------

width = 20
height = 20
DIM = width * height
room = np.zeros((width,height))
environment = np.zeros((width,height))
bulbs = np.zeros((width,height))
presence = presence_scenario_a(width, height)

#----- Mape


def fitness(individual, data):
    print (data)
    print (individual)
    match = 0
    for x in range(width):
        for y in range(height):
            if presence[x,y]>0 and individual[x,y]>0:
                match += 1
    match_percentage = match / width * height 
    return match_percentage

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("random_num", random.choice, [0,1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_num, n=DIM)
print(toolbox.individual())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluateInd(individual):
    match = 0
    for y in range(height): # top left is (X=0,Y=0)
        for x in range(width):
            if individual[y*width+x]>0 and presence[x,y]>0:
                match += 1
    match_percentage = match / (width * height) 
    return (float(match_percentage),)


def evaluateInd2(individual):
    match = 0
    for y in range(height): # top left is (X=0,Y=0)
        for x in range(width):
            if individual[y*width+x] == presence[x, y]:
                match += 1
    match_percentage = match / (width * height) 
    return (float(match_percentage),)


def myMutation(individual):
    for i in range(width):
        individual[random.randint(0,len(individual)-1)] = random.choice([0,1])
    return (individual,)
    
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", evaluateInd2)
toolbox.register("mutate", myMutation)

def decode(individual):
    bulbs = np.zeros((width,height))
    for y in range(height): # top left is (X=0,Y=0)
        for x in range(width):
            bulbs[x,y] = individual[y*width+x]
    return bulbs
 
fig=plt.figure(figsize=(1, 2))

fig.add_subplot(121)
plt.imshow(presence, cmap='gray', interpolation='nearest', vmin=0, vmax=1)

fig.add_subplot(122)
im = plt.imshow(bulbs, cmap='gray', interpolation='bilinear', animated=True, vmin=0, vmax=1)
pop = toolbox.population(n=1000)

def updatefig(*args):
    algorithms.eaMuPlusLambda (
            pop, toolbox, 
            400, 100, #parents, children
            0.8, 0.2, #probabilities
            1) #iterations

    top = sorted(pop, key=lambda x:x.fitness.values[0])[-1]
    fit = top.fitness.values[0]
    print ('fitness-: {}'.format(fit))
        
    im.set_data(decode(top))
    #im.set_cmap("gray")
    #im.update()
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()

#if __name__ == "__main__": 
#    
#    pop = toolbox.population(n=1000)
#  
#    fit = 0.0
#    while (fit < 0.95):
#  
#        algorithms.eaMuPlusLambda (
#            pop, toolbox, 
#            400, 100, #parents, children
#            0.8, 0.2, #probabilities
#            1) #iterations
#    
#        top = sorted(pop, key=lambda x:x.fitness.values[0])[-1]
#        fit = top.fitness.values[0]
#        print ('fitness: {}'.format(fit))
#        
#    
#    print (top)
#------- visualization

#fig=plt.figure(figsize=(2, 2))
#
#fig.add_subplot(221)
#plt.imshow(room, cmap='gray', interpolation='nearest')
#
#fig.add_subplot(222)
#plt.imshow(environment, cmap='gray', interpolation='nearest')
#
#fig.add_subplot(223)
#plt.imshow(decode(top), cmap='gray', interpolation='nearest')
#
#fig.add_subplot(224)
#plt.imshow(presence, cmap='gray', interpolation='nearest')




