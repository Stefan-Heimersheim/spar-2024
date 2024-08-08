# %%
# Imports
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import DeadFeaturesOracle


# %%
dead_features_oracle = DeadFeaturesOracle()

dead_features_oracle.is_dead(3, 456)
