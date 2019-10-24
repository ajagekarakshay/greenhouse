import os
import pickle

import numpy as np
import scipy as sp

ANALYSIS_FOLDER = '/Users/pihatonttu/git/self-aware-bulb/r1'

PICKLE_FILES = [os.path.join(ANALYSIS_FOLDER, a) for a in os.listdir(ANALYSIS_FOLDER) if a.endswith(".pkl")]
print(PICKLE_FILES)

STATS = {'S11': {}, 'S12': {}, 'S21': {}, 'S22': {}, 'S31': {}, 'S32': {}}
CONTEXTS = 2001

for pkl in PICKLE_FILES:
    with open(pkl, 'rb') as f:
        stats = pickle.load(f)
        for name, values in stats.items():
            #print("Statistics for '{}':".format(name))
            half = int(CONTEXTS / 2)
            tnth = int(CONTEXTS / 10)
            if name == "S1":
                first = "S11"
                last = "S12"
            if name == "S2":
                first = "S21"
                last = "S22"
            if name == "S3":
                first = "S31"
                last = "S32"

            for k, v in values.items():
                if len(v) > 0 and k not in ['tops', 'plan', 'lux']:
                    tenth = v[0][:half]
                    if k not in STATS[first]:
                        STATS[first][k] = [tenth]
                    else:
                        STATS[first][k].append(tenth)
                    tenth = v[0][half:]
                    if k not in STATS[last]:
                        STATS[last][k] = [tenth]
                    else:
                        STATS[last][k].append(tenth)

print("{:*>100}".format(""))
print("{:*>100}".format(""))
print("{:*>100}".format(""))

for name, values in STATS.items():
    print()
    print("Statistics for '{}':".format(name))
    for k, v in values.items():
        print("{:10s} AVG {:9.3f} | STD {:9.3f} | MIN {:9.3f} | MAX {:9.3f}".format("{}".format(k),
                                                                                    np.mean(v),
                                                                                    np.std(v),
                                                                                    np.min(v),
                                                                                    np.max(v)))
