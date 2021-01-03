import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns; sns.set()
import pandas as pd
from matplotlib import animation, rc
from collections import namedtuple
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from bokeh.io import show, output_notebook
from bokeh.palettes import PuBu4
from bokeh.plotting import figure
from bokeh.models import Label
from qbstyles import mpl_style
mpl_style(dark=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 1.00
TARGET_UPDATE = 20
BATCH_SIZE = 512

# Environment parameters
# Demand formula --> price = intercept (q) + slope (k) * quantity
# p = 50 + -0.5 * 50
T = 20                # Time steps in the price schedule
price_max = 200       # Maximum valid price, dollars
price_step = 5       # Minimum valid price change, dollars
q_0 = 105            # Intercept in the demand function q_t
k = 0.16                # Slope in the demand function, q_t
unit_cost = 50       # Product production cost, dollars
a_q = 3000             # Response coefficient for price increase
b_q = 300             # Response coefficient for price decrease

price_grid = np.arange(price_step, price_max, price_step)
price_change_grid = np.arange(0.5, 2.0, 0.1)
profit_map = np.zeros((len(price_grid), len(price_change_grid)))
