import os
import pickle
import random
import sys

import numpy as np

from greenhouse.creative_greenhouse import CreativeGreenhouse


SET_SEEDS = True

if SET_SEEDS:
    random.seed(0)
    np.random.seed(0)


def figure_situation(height, width):
    """Create situation for paper figure.
    """
    mean = 0.50
    std = (random.random() * 0.2) + 0.1
    situation = np.zeros((height, width)) + mean + np.random.normal(0, std, (height, width))
    situation[situation < 0] = 0.0
    situation[situation > 1] = 1.0
    ambient_lux = 5000
    return {'tops': situation, 'std': std, 'mean': mean, 'ambient_lux': ambient_lux}


def create_situations(num, height, width, means=None, ambient_lux=None):
    """Create a fixed set of situations which can be used for each system.
    """
    means = [0.35, 0.5, 0.65] if means is None else means
    ambient_lux = [0.0, 500.0, 1000.0, 3000.0, 5000.0, 7000.0, 9000.0] if ambient_lux is None else ambient_lux

    situations = []
    for n in range(num):
        s = {}
        std = (random.random() * 0.2) + 0.1
        mean = random.choice(means)
        ambient = random.choice(ambient_lux)
        situation = np.zeros((height, width)) + mean + np.random.normal(0, std, (height, width))
        situation[situation < 0] = 0.0
        situation[situation > 1] = 1.0
        s['tops'] = situation
        s['std'] = std
        s['mean'] = mean
        s['ambient_lux'] = ambient
        situations.append(s)

    return situations

WIDTH = 20
HEIGHT = 20
RUNS_PER_CONFIG = 1
MAX_STEPS = 10
POP_SIZE = 40
PARENTS = 100
CONTEXTS_PER_CONFIG = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SHOW_ANIMATION = False
SAVE_PREFIX = sys.argv[1] if len(sys.argv) > 1 else "gh-run"
print(SAVE_PREFIX, CONTEXTS_PER_CONFIG)
SAVE_FOLDER = "{}-{}-{}-{}-{}-{}".format(SAVE_PREFIX, RUNS_PER_CONFIG, CONTEXTS_PER_CONFIG, MAX_STEPS, POP_SIZE, PARENTS)
SAVE_IMAGES = False
IMAGE_SAVE_FOLDER = SAVE_FOLDER if SAVE_IMAGES else None
stats = {}

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# each config is 2-tuple: (name, used awarenesses), where used awarenesses is 4-tuple of booleans in order:
# goal, resource, context and time
awareness_configs = [
    ('S3', [True, True, True, True]),
    ('S2', [True, False, True, True]),
    ('S1', [True, False, False, True]),
]

FIG_SIT = figure_situation(HEIGHT, WIDTH)
SITUATIONS = create_situations(CONTEXTS_PER_CONFIG, HEIGHT, WIDTH)

# Run each config
for config in awareness_configs:
    name = config[0]
    all_pickle_file = os.path.join(SAVE_FOLDER, "stats_{}_all.pkl".format(name))
    awarenesses = config[1]
    stats[name] = {'fitness': [], 'time': [], 'luxavg': [], 'luxstd': [], 'luxmin': [], 'luxmax': [], 'bulbs': [],
                   'sceavg': [], 'scestd': [], 'luxamb': [], 'tops': [], 'plan': [], 'lux':[]}

    for r in range(RUNS_PER_CONFIG):
        print("RUN {}/{} FOR CONFIG '{}':".format(r+1, RUNS_PER_CONFIG, name))
        pickle_file = os.path.join(SAVE_FOLDER, "stats_{}_{}.pkl".format(name, r))
        creative_home = CreativeGreenhouse(WIDTH, HEIGHT, awarenesses, pickle_file=pickle_file,
                                           show_ani=SHOW_ANIMATION, deap_config={'max_steps': MAX_STEPS,
                                                                                 'pop_size': POP_SIZE,
                                                                                 'parents': PARENTS})

        if SHOW_ANIMATION:
            cur_stats = creative_home.run_animation()
        else:
            cur_stats = creative_home.run_multiple_contexts(num_contexts=CONTEXTS_PER_CONFIG,
                                                            situations=SITUATIONS + [FIG_SIT],
                                                            img_save_folder=IMAGE_SAVE_FOLDER,
                                                            img_prefix="{}_r{}".format(name, r+1),
                                                            title_prefix=name)

        if cur_stats is not None:
            for k, v in cur_stats.items():
                stats[name][k].append(v)

            print()
            print("Current run statistics:")
            for k, v in cur_stats.items():
                if len(v) > 0 and k not in ['tops', 'plan', 'lux']:
                    print(
                        "{:10s} AVG {:9.3f} | STD {:9.3f} | MIN {:9.3f} | MAX {:9.3f}".format(k, np.mean(v), np.std(v),
                                                                                              np.min(v), np.max(v)))
            print()

        if r == RUNS_PER_CONFIG - 1:
            with open(all_pickle_file, 'wb') as f:
                pickle.dump(stats[name], f)

        print()

# Save statistics
pickle_file = os.path.join(SAVE_FOLDER, "stats.pkl")
with open(pickle_file, 'wb') as f:
    pickle.dump(stats, f)


# Print main statistics
print("{:*>100}".format(""))

for name, values in stats.items():
    print("All statistics for '{}':".format(name))
    for k, v in values.items():
        if len(v) > 0 and k not in ['tops', 'plan', 'lux']:
            print("{:10s} AVG {:9.3f} | STD {:9.3f} | MIN {:9.3f} | MAX {:9.3f}".format(k, np.mean(v), np.std(v), np.min(v), np.max(v)))
    print()

print("{:*>100}".format(""))
for name, values in stats.items():
    print("Last tenth statistics for '{}':".format(name))
    last_tenth = int(CONTEXTS_PER_CONFIG / 10)
    for k, v in values.items():
        if len(v) > 0 and k not in ['tops', 'plan', 'lux']:
            tenth = v[0][-last_tenth:]
            print("{:10s} AVG {:9.3f} | STD {:9.3f} | MIN {:9.3f} | MAX {:9.3f}".format("{}/10".format(k), np.mean(tenth), np.std(tenth), np.min(tenth), np.max(tenth)))
    print()

print("{:*>100}".format(""))

##########################################################################################
#######################  PLOTS PLOTS PLOTS  ##############################################
##########################################################################################

if SAVE_IMAGES:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create figure from FIG_SIT
    names = [a[0] for a in awareness_configs]
    tops = stats[names[0]]['tops'][0][-1]

    fig = plt.figure(figsize=(15.5, 3.0))
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=0.0, hspace=0.0)

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(14.5, 3), ncols=4)

    ax = plt.subplot(gs1[0])
    title = 'Plant heights'
    ax.set_title(title, y=1.00, fontsize=14)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    bbox = ax.get_position()
    print(bbox)
    image = ax.imshow(tops, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0, animated=False)
    cbaxes = fig.add_axes([0.124, 0.11, 0.01, 0.77])
    #cbaxes.set_xticks([])
    cbar = plt.colorbar(image, cax=cbaxes)
    cbar.ax.yaxis.set_ticks_position('left')

    idx = 2
    ax = plt.subplot(gs1[1])
    name = names[idx]
    title = '{}: Lux & Lamps ON'.format(name)
    ax.set_title(title, y=1.00, fontsize=14)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    lux = stats[name]['lux'][0][-1]
    lux_img = ax.imshow(lux, cmap='inferno', interpolation='nearest', vmin=0.0, vmax=15000.0, animated=False)
    #fig.colorbar(lux_img, ax=ax)

    bulbs_on = stats[name]['plan'][0][-1]
    print("{} {}".format(name, np.count_nonzero(bulbs_on)))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if bulbs_on[i, j] > 0:
                text = ax.text(j, i, "+", ha="center", va="center", color="black", fontsize=12)

    idx = 1
    ax = plt.subplot(gs1[2])
    name = names[idx]
    title = '{}: Lux & Lamps ON'.format(name)
    ax.set_title(title, y=1.00, fontsize=14)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    lux = stats[name]['lux'][0][-1]
    lux_img = ax.imshow(lux, cmap='inferno', interpolation='nearest', vmin=0.0, vmax=15000.0, animated=False)
    #fig.colorbar(lux_img, ax=ax)

    bulbs_on = stats[name]['plan'][0][-1]
    print("{} {}".format(name, np.count_nonzero(bulbs_on)))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if bulbs_on[i, j] > 0:
                text = ax.text(j, i, "+", ha="center", va="center", color="black", fontsize=12)

    idx = 0
    ax = plt.subplot(gs1[3])
    name = names[idx]
    title = '{}: Lux & Lamps ON'.format(name)
    ax.set_title(title, y=1.00, fontsize=14)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    bbox = ax.get_position()
    print(bbox)
    lux = stats[name]['lux'][0][-1]
    lux_img = ax.imshow(lux, cmap='inferno', interpolation='nearest', vmin=0.0, vmax=15000.0, animated=False)
    cbaxes = fig.add_axes([0.90, 0.11, 0.01, 0.77])
    #cbaxes.set_xticks([])
    plt.colorbar(lux_img, cax=cbaxes, extend='max')

    bulbs_on = stats[name]['plan'][0][-1]
    print("{} {}".format(name, np.count_nonzero(bulbs_on)))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if bulbs_on[i, j] > 0:
                text = ax.text(j, i, "+", ha="center", va="center", color="black", fontsize=12)

    #plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, "paper_figure.pdf"))






