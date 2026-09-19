import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .visual_kiwi_env import PRIVILEGED_KEYS, assert_actor_observation


class ActorVisualExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        leaked = set(observation_space.spaces).intersection(PRIVILEGED_KEYS)-{'privileged'}
        if leaked:
            raise AssertionError(f'Privileged keys in actor observation space: {sorted(leaked)}')
        self.encoders = nn.ModuleDict()
        for name in ('rgb', 'depth'):
            channels = observation_space[name].shape[0]
            self.encoders[name] = nn.Sequential(
                nn.Conv2d(channels, 24, 5, stride=2), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
                nn.Flatten())
        with torch.no_grad():
            image_features = sum(self.encoders[name](torch.zeros(1, *observation_space[name].shape)).shape[1]
                                 for name in self.encoders)
        extras = ('proprio', 'phase', 'has_grasped', 'has_placed', 'age')
        inputs = image_features+sum(observation_space[name].shape[0] for name in extras)
        self.fusion = nn.Sequential(nn.Linear(inputs, features_dim), nn.ReLU())

    def forward(self, observations):
        actor = {name: observations[name] for name in ('rgb', 'depth', 'proprio', 'phase', 'has_grasped', 'has_placed', 'age')}
        assert_actor_observation(actor)
        encoded = [self.encoders[name](observations[name]) for name in self.encoders]
        extras = [observations[name] for name in ('proprio', 'phase', 'has_grasped', 'has_placed', 'age')]
        return self.fusion(torch.cat([*encoded, *extras], dim=1))


class CriticVisualExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        width = observation_space['privileged'].shape[0]+observation_space['proprio'].shape[0]
        self.mlp = nn.Sequential(nn.Linear(width, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.mlp(torch.cat([observations['privileged'], observations['proprio']], dim=1))


class VisualKiwiExtractor(ActorVisualExtractor):
    """Back-compat alias used by older trainer kwargs; actor-only."""


class AsymmetricVisualPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['share_features_extractor'] = False
        self._extractors_built = 0
        super().__init__(*args, **kwargs)

    def make_features_extractor(self):
        self._extractors_built += 1
        kwargs = dict(self.features_extractor_kwargs)
        if self._extractors_built == 1:
            return ActorVisualExtractor(self.observation_space, **kwargs)
        return CriticVisualExtractor(self.observation_space, **kwargs)
