import pandas as pd
import numpy as np
import warnings
# import lightgbm as lgb
from core.etc.base_definitions import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


class DemandForecast:
    def __init__(self):
        self.train = pd.read_csv(TRAIN_CSV, parse_dates=['date'])
        self.test = pd.read_csv(TEST_CSV, parse_dates=['date'])
        self.sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)

        self.df = pd.concat([self.train, self.test], sort=False)

        print(self.df)
