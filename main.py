from smart_home import SmartHome
from v2.control_home import ControlHome
from v2.creative_home import CreativeHome
from v2.adaptive_home import AdaptiveHome
import sys

WIDTH = 20
HEIGHT = 20

for arg in sys.argv:
    if arg == 'control':
        control_home = ControlHome(WIDTH, HEIGHT)
        control_home.run()
    elif arg == 'adaptive':
        adaptive_home = AdaptiveHome(WIDTH, HEIGHT) #simple self-adaptive home, uses strategies
        adaptive_home.run()
        # complex self-adaptive home, uses evolutionary algo
        #smart_home = SmartHome(WIDTH,HEIGHT)
        #smart_home.run()
    elif arg == 'creative':
        creative_home = CreativeHome(WIDTH,HEIGHT)
        creative_home.run()




