from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np
from .linear_feature_baseline import LinearFeatureBaseline


class InfoLinearFeatureBaseline(LinearFeatureBaseline):

    @overrides
    def _features(self, path):
        o = np.clip(path["env_infos"]["baseline_obs"]["feat"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

