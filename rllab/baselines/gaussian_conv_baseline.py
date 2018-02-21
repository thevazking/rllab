import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.regressors.gaussian_conv_regressor import GaussianConvRegressor


class GaussianConvBaseline(Baseline, Parameterized):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            only_first_dimension=False,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianConvBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self.only_first_dimension = only_first_dimension
        print("env observation space")
        print(env_spec.observation_space.shape)

        if self.only_first_dimension:
            shape = (1,) + env_spec.observation_space.shape[1:]
            self._regressor = GaussianConvRegressor(
                input_shape=shape,
                output_dim=1,
                name="vf",
                **regressor_args
            )
        else:
            self._regressor = GaussianConvRegressor(
                input_shape=env_spec.observation_space.shape,
                output_dim=1,
                name="vf",
                **regressor_args
            )

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        print("observations shape in fit")
        print(observations.shape)
        if self.only_first_dimension:
            observations = observations[:,0:1,...]
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        observations = path['observations']
        print("observations shape in predict")
        print(observations.shape)
        if self.only_first_dimension:
            observations = observations[:,0:1,...]
        return self._regressor.predict(observations).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
