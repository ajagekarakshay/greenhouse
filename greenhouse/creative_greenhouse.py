"""Creatively self-adapting greenhouse light control system.

See 'run_greenhouse_simulations' for usage.
"""
import logging
import pickle
import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from deap import base, creator, algorithms, benchmarks, tools

import random
from greenhouse.grow_scenario import GrowScenario
from greenhouse.plan import Plan
import time


class CreativeGreenhouse:

    def __init__(self, width, height, awarenesses=[True, True, True, True], pickle_file='stats.pkl',
                 show_ani=True, deap_config={}):
        self.width = width
        self.height = height
        self.scenario = GrowScenario(self.width, self.height)
        self.scenario.reinitialize()
        self.bulbs_on = np.zeros((self.height, self.width))

        self.TARGET_LUX = 10000
        self.LUX_FACTOR = scipy.stats.norm.pdf(1, 1, 0.2)
        self.PLANT_HEIGHT_ESTIMATE = 0.50
        self.plant_tops_estimate = np.zeros((self.height, self.width)) + self.PLANT_HEIGHT_ESTIMATE

        self.STEPS = 0
        self.MAX_STEPS = deap_config['max_steps']
        self.POP_SIZE = deap_config['pop_size']
        self.PARENTS = deap_config['parents']
        self.DIM = self.width * self.height

        self.goal_aware = awarenesses[0]
        self.resource_aware = awarenesses[1]
        self.context_aware = awarenesses[2]
        self.time_aware = awarenesses[3]

        self.goals = {
            'luminosity': {
                'enabled': True,
                'maximize': True,
                'evaluate': self.evaluate_lux
            },
            'cost': {
                'enabled': True if self.resource_aware else False,
                'maximize': False,
                'evaluate': self.evaluate_cost
            }
        }

        self.BULB_COST_FACTOR = 0.4

        self.show_ani = show_ani
        self.stats = {'fitness': [], 'time': [], 'luxavg': [], 'luxstd': [], 'luxmin': [], 'luxmax': [], 'bulbs': [],
                      'sceavg': [], 'scestd': [], 'luxamb': [], 'tops': [], 'plan': [], 'lux': []}
        self.pickle_file = pickle_file

        if self.show_ani:
            self.init_figures()
        else:
            self.init_figures2()

        # Initialize DEAP
        self.toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        self.toolbox.register("random_num", random.choice, [0, 1])
        self.toolbox.register("individual", self.generate_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("select", tools.selBest)
        #self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self.fitness_ea)
        self.toolbox.register("mutate", self.mutate)
        self.pop = []
        self.starting_plans = self.create_starting_plans()
        self.plans = [p for p in self.starting_plans]
        self.initialize_population()

        self.starting_time = time.monotonic()

    def create_starting_plans(self):
        """Create some base plans which are "envisioned by the designer".
        """
        ctxs = [np.zeros((self.height, self.width)),
                np.zeros((self.height, self.width)) + 0.2,
                np.zeros((self.height, self.width)) + 0.35,
                np.zeros((self.height, self.width)) + 0.5,
                np.zeros((self.height, self.width)) + 0.65,
                np.zeros((self.height, self.width)) + 0.80]
        ambient_lux = [0.0, 500.0, 1000.0, 3000.0, 5000.0, 7000.0, 9000.0]

        plans = []

        # Create sparse grid plans (and empty and full grid plans)
        for i in range(0, 6):
            bulbs_on = np.zeros((self.width, self.height))
            if i > 0:
                bulbs_on[::i, ::i] = 1
            cost = self.scenario.compute_cost(bulbs_on)
            ind = self.toolbox.individual(bulbs_on.flatten())
            plan = Plan(ind, cost)

            for ctx in ctxs:
                for lux in ambient_lux:
                    fit = self.fitness_ea(ind, tops=ctx)
                    plan.add_applied_context(ctx, self.scenario.bulbs_height, fit, lux)
            plans.append(plan)

        # Create dense grid plans
        for i in range(2, 5):
            bulbs_on = np.zeros((self.width, self.height))
            bulbs_on[::i, :] = 1
            bulbs_on[:, ::i] = 1
            cost = self.scenario.compute_cost(bulbs_on)
            ind = self.toolbox.individual(bulbs_on.flatten())
            plan = Plan(ind, cost)

            for ctx in ctxs:
                for lux in ambient_lux:
                    fit = self.fitness_ea(ind, tops=ctx)
                    plan.add_applied_context(ctx, self.scenario.bulbs_height, fit, lux)
            plans.append(plan)

        return plans

    def update_individual (self, individual, data):
        individual[:] = data.flatten()

    def encode(self, individual):
        return individual.flatten()

    def decode(self, individual):
        return individual.reshape((self.width, self.height))

    def generate_individual(self, icls, content=None):
        """Generate new DEAP individual.
        """
        if content is None:
            bulbs_on = np.zeros((self.height, self.width))
            bulbs_on[np.random.random((self.height, self.width)) > 0.5] = 1
            #bulbs_on[self.scenario.bulbs == -1] = 0
            genome = self.encode(bulbs_on)
            individual = icls(genome)
        else:
            individual = icls(content)
        return individual

    def mutate(self, individual):
        """Mutate a DEAP individual.
        """
        bulbs_on = self.decode(individual)
        x = np.random.randint(0, self.width, (self.width,))
        y = np.random.randint(0, self.height, (self.width,))
        # Flip the bits on mutate
        bulbs_on[y, x] = 1 - bulbs_on[y, x]
        #bulbs_on[y, x] = np.random.randint(0, 2, (self.width,))
        self.update_individual(individual, bulbs_on)
        return individual,

    # ------- main fitness functions -------

    def fitness_ea(self, individual, tops=None):
        """Main fitness function to compute fitness of a DEAP individual.
        """
        bulbs = self.decode(individual)
        #return float(self.fitness(bulbs, tops=tops)), self.evaluate_cost(bulbs)
        return float(self.fitness(bulbs, tops=tops)),

    def fitness(self, plan, tops=None):
        fitness = 0

        if self.goal_aware:
            for goal_name, goal in self.goals.items():
                if goal['enabled']:
                    if goal['maximize']:
                        fitness += goal['evaluate'](plan, tops=tops)
                    else:
                        fitness -= goal['evaluate'](plan, tops=tops)

        #if self.resource_aware:
        #    fitness += self.evaluate_resource(plan)

        #if self.domain_aware and self.context_aware:
        #    fitness += self.evaluate_domain_context(plan)

        return fitness

    # ------- evaluation of awarenesses
    def evaluate_lux(self, plan, tops=None):
        """Evaluate fitness of the plan w.r.t. lux target.
        """
        if tops is None and self.context_aware:
            tops = self.scenario.tops
        elif not self.context_aware:
            tops = self.plant_tops_estimate
        lux = self.scenario.compute_lux(tops=tops, bulbs=plan)
        score = np.sum(scipy.stats.norm.pdf((lux / self.TARGET_LUX), 1, 0.2) / self.LUX_FACTOR)
        return score

    def evaluate_cost(self, plan, tops=None):
        """Cost is based on the number of turned on lamps.
        """
        return np.count_nonzero(plan) * self.BULB_COST_FACTOR

    def hof_eq(self, a1, a2):
        if np.any(a1 != a2):
            return False
        return True

    def new_context(self, situation=None):
        if situation is None:
            self.scenario.reinitialize()
        else:
            self.scenario.reinitialize(**situation)
        self.bulbs_on = np.zeros((self.height, self.width))

    def initialize_population(self, use_old_plans=True, use_closest_plans=True):
        if not use_old_plans:
            self.pop = self.toolbox.population(n=self.POP_SIZE)
        else:
            new_inds = int(max(self.POP_SIZE / 2, self.POP_SIZE - len(self.plans)))
            logging.debug("Creating {} new individuals".format(new_inds))
            if new_inds > 0:
                self.pop = self.toolbox.population(n=new_inds)
            else:
                self.pop = []

            logging.debug("Adding {} old plans to population".format(self.POP_SIZE - new_inds))
            if use_closest_plans:
                old_plans = self.get_closest_tops_plans(self.scenario.presence, self.POP_SIZE - new_inds)
            else:
                old_plans = random.sample(self.plans, min(len(self.plans), int(self.POP_SIZE / 2)))
            for p in old_plans:
                individual = self.toolbox.individual(p.bulbs_on)
                self.pop.append(individual)

    def get_closest_tops_plans(self, tops, number_of_plans=1):
        t1 = time.monotonic()
        if len(self.plans) < number_of_plans:
            return self.plans

        plans = []
        for p in self.plans:
            min_dist = np.iinfo(np.int64).max
            for ctx in p.applied_contexts:
                if abs(ctx.ambient_lux - self.scenario.ambient_lux) < 1500:
                    eucl_dist = np.linalg.norm(tops - ctx.tops)
                    if eucl_dist < min_dist:
                        min_dist = eucl_dist
            plans.append((min_dist, p))

        plans = sorted(plans, key=lambda x: x[0])
        logging.debug(plans)
        logging.debug("Closest tops plans found in {:.4f} seconds.".format(time.monotonic() - t1))
        return [p[1] for p in plans[:number_of_plans]]

    def get_best_plans(self, tops, ambient_lux, number_of_plans=1):
        return []

    # ----------  Simulation ----------
    def run_animation(self):
        if self.show_ani:
            self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=50, blit=True)
            plt.show()

    def run_single_context(self, img_save_path=None, title_prefix=None):
        algorithms.eaMuPlusLambda(self.pop, self.toolbox,
                                  self.PARENTS, self.POP_SIZE,
                                  0.9, 0.1, self.MAX_STEPS,
                                  verbose=False)

        top_plan = sorted(self.pop, key=lambda x: x.fitness.values[0])[-1]
        fit = top_plan.fitness.values[0]

        self.bulbs_on = self.decode(top_plan)
        plan_bulbs = np.copy(top_plan)

        tops = self.scenario.presence
        if not self.context_aware:
            tops = self.plant_tops_estimate

        plan = Plan(plan_bulbs, cost=self.scenario.compute_cost(bulbs=plan_bulbs))
        plan.add_applied_context(np.copy(tops),
                                 np.copy(self.scenario.bulbs_height),
                                 fit, self.scenario.ambient_lux)
        self.plans.append(plan)

        if img_save_path is not None:
            self.init_figures2(title_prefix)
            self.fig.savefig(img_save_path)
            plt.close()

        return plan, fit

    def run_multiple_contexts(self, num_contexts=10, situations=None, img_save_folder=None, img_prefix="", title_prefix=""):
        num_contexts = num_contexts if situations is None else len(situations)

        for n in range(num_contexts):
            t1 = time.monotonic()

            # Create new context, initialize population and optimize for it
            situation = None
            if situations is not None:
                situation = situations[n]
            self.new_context(situation)
            self.initialize_population()
            if img_save_folder is not None and (n == 0 or (n+1) % 10 == 0):
                img_path = os.path.join(img_save_folder, "{}_{}.pdf".format(img_prefix, n+1))
            plan, fitness = self.run_single_context(img_path, title_prefix)

            lux = self.scenario.compute_lux(bulbs=self.bulbs_on)
            num_bulbs = np.count_nonzero(self.bulbs_on)
            luxavg = np.mean(lux)
            luxstd = np.std(lux)
            luxmin = np.min(lux)
            luxmax = np.max(lux)
            t2 = time.monotonic()

            self.stats['fitness'].append(fitness)
            self.stats['time'].append(t2 - t1)
            self.stats['luxavg'].append(luxavg)
            self.stats['luxstd'].append(luxstd)
            self.stats['luxmin'].append(luxmin)
            self.stats['luxmax'].append(luxmax)
            self.stats['bulbs'].append(num_bulbs)
            self.stats['sceavg'].append(self.scenario.mean)
            self.stats['scestd'].append(self.scenario.std)
            self.stats['luxamb'].append(self.scenario.ambient_lux)
            self.stats['tops'].append(self.scenario.tops)
            self.stats['plan'].append(np.copy(self.bulbs_on))
            self.stats['lux'].append(lux)

            print('CTX {:5d} ({:4.0f} {:.2f} {:.3f})| FIT {:7.3f} (AVG {:7.3f}) | LUX AVG {:9.3f} STD {:8.3f} MIN {:9.3f} MAX {:9.3f} | #BULBS {:3d} (AVG {:7.3f}) | PLANS {:5d} ({:6.3f}s)'
                  .format(n+1, self.scenario.ambient_lux, self.scenario.mean, self.scenario.std, fitness, np.mean(self.stats['fitness']), luxavg, luxstd, luxmin, luxmax, num_bulbs,
                          np.mean(self.stats['bulbs']), len(self.plans), t2 - t1))

        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.stats, f)
        return self.stats

    #------------------ simulation loop ----------------
    def updatefig(self, *args):
        t1 = time.monotonic()
        # Run one generation of the EA algorithm
        self.initialize_population()
        algorithms.eaMuPlusLambda(self.pop, self.toolbox, self.PARENTS, self.POP_SIZE, 0.9, 0.1, 1, verbose=False)
        top_plan = sorted(self.pop, key=lambda x: x.fitness.values[0])[-1]
        fitness = top_plan.fitness.values[0]

        self.bulbs_on = self.decode(top_plan)
        plan_bulbs = np.copy(top_plan)
        plan = Plan(plan_bulbs, cost=self.scenario.compute_cost(bulbs=plan_bulbs))
        plan.add_applied_context(np.copy(self.scenario.presence),
                                 np.copy(self.scenario.bulbs_height),
                                 fitness,
                                 self.scenario.ambient_lux)
        self.plans.append(plan)

        lux = self.scenario.compute_lux(bulbs=self.bulbs_on)
        num_bulbs = np.count_nonzero(self.bulbs_on)
        luxavg = np.mean(lux)
        luxstd = np.std(lux)
        t2 = time.monotonic()
        print('GEN {:5d} | FIT {:7.3f} | LUX AVG {:.3f} STD {:.3f} | #BULBS {} | PLANS {} ({:.3f}s)'
              .format(self.STEPS, fitness, np.mean(lux), np.std(lux), num_bulbs, len(self.plans), t2 - t1))

        # Add bookkeeping of needed stats
        self.stats['fitness'].append(fitness)
        self.stats['time'].append(t2 - t1)
        self.stats['luxavg'].append(luxavg)
        self.stats['luxstd'].append(luxstd)
        self.stats['bulbs'].append(num_bulbs)

        self.STEPS += 1
        if self.STEPS > self.MAX_STEPS:
            self.ani.event_source.stop()  # it takes a while for the animation loop to notice the event
        else:
            self.bulb_img.set_data(self.bulbs_on)
            self.lux_img.set_data(lux)
        return self.bulb_img, self.lux_img

    # Plots and figures
    def create_plot(self, ax, title, data, plot_cmap='gray_r', plot_interpolation='nearest',
                    plot_vmin=0, plot_vmax=1, plot_animated=False, alpha=1.0):
        ax.set_title(title, y=1.00)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        image = ax.imshow(data, cmap=plot_cmap, interpolation=plot_interpolation,
                          vmin=plot_vmin, vmax=plot_vmax, animated=plot_animated, alpha=alpha)
        if title == 'Lux' or title == 'Plant tops':
            self.fig.colorbar(image, ax=ax)
        return image

    def init_figures(self):
        self.fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 3), ncols=3)
        self.create_plot(ax1, title='Plant heights', data=self.scenario.presence,
                         plot_cmap='gray', plot_interpolation='nearest',
                         plot_vmin=0, plot_vmax=1, plot_animated=False)
        self.bulb_img = self.create_plot(ax2, title='Bulbs on', data=self.bulbs_on,
                                         plot_cmap='Blues', plot_interpolation='nearest',
                                         plot_vmin=0, plot_vmax=1, plot_animated=True, alpha=1.0)
        self.lux_img = self.create_plot(ax3, title='Lux', data=self.scenario.compute_lux(bulbs=self.bulbs_on),
                                        plot_cmap='inferno', plot_interpolation='nearest',
                                        plot_vmin=0, plot_vmax=15000, plot_animated=True)
        plt.tight_layout()

    def init_figures2(self, title_prefix=None):
        self.fig, (ax1, ax2) = plt.subplots(figsize=(7.5, 3), ncols=2)

        ax = ax1
        title = 'Plant heights'
        ax.set_title(title, y=1.00, fontsize=16)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        image = ax.imshow(self.scenario.tops, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0, animated=False)
        self.fig.colorbar(image, ax=ax)

        ax = ax2
        title = 'Lux and Lamps ON (+)'
        if title_prefix is not None:
            title = "{}: {}".format(title_prefix, title)
        ax.set_title(title, y=1.00, fontsize=16)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        lux = self.scenario.compute_lux(bulbs=self.bulbs_on)
        self.lux_img = ax.imshow(lux, cmap='inferno', interpolation='nearest', vmin=0.0, vmax=15000.0, animated=False)
        self.fig.colorbar(self.lux_img, ax=ax)

        for i in range(self.height):
            for j in range(self.width):
                if self.bulbs_on[i, j] > 0:
                    text = ax.text(j, i, "+", ha="center", va="center", color="black", fontsize=12)

        plt.tight_layout()

