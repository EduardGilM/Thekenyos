"""SI material presets. Provenance and transfer limits: docs/kiwi-material-evidence.md.

Hayward geometry/stems and Xuxiang tissue are separate experiments. Unknown
contact/damage laws remain explicit proxies, never measured cultivar values.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Tissue:
    young: float
    poisson: float
    yield_stress: float
    tangent: float


XUXIANG = {
    'flesh': Tissue(1.57e6, .4, .26e6, .92e6),
    'skin': Tissue(10.69e6, .3, .53e6, 0.),
    'core': Tissue(5.11e6, .4, 1.12e6, .83e6),
}
# He2024 means; stiffness of stem tissue, not abscission strength.
STEM_LENGTH = .04384
STEM_DIAMETER = .00368
STEM_YOUNG = 325e6
STEM_AREA = np.pi * STEM_DIAMETER**2 / 4
STEM_I = np.pi * STEM_DIAMETER**4 / 64
STEM_AXIAL = STEM_YOUNG * STEM_AREA / STEM_LENGTH
STEM_BENDING = 3 * STEM_YOUNG * STEM_I / STEM_LENGTH**3
# Mu2020 pooled forces. Uniform sampling is engineering domain randomization.
DETACH_RANGE = (1.08, 12.25)
RUBBER_STATIC_RANGE = (.38, .51)
# Unknown actual liner/fruit contacts; deliberately not attributed to rubber.
FRUIT_FRICTION = .6
LINER_FRICTION = .7
RESTITUTION = .05


def sample_hayward(rng):
    """Rejection sample coupled Mu2020 size/mass and Razavi2007 density bounds.

    Population intersection is a proposed robustness envelope, not a measured
    joint distribution. Axial length fixed at Mu's reported mean.
    """
    for _ in range(10000):
        radii = np.array([rng.uniform(.04270, .05184),
                          rng.uniform(.04585, .05713), .06498]) / 2
        density = rng.uniform(940., 1040.)
        mass = 4*np.pi*np.prod(radii)*density/3
        if .0814 <= mass <= .1287:
            return radii, float(mass)
    raise RuntimeError('Hayward mass/geometry rejection sampler failed')


def damage_increment(strain, stress, dt, damage):
    """Irreversible diagnostic proxy, NOT a calibrated bruise probability.

    5% comes from slow whole-fruit Xuxiang plates, .26MPa from flesh coupons.
    Integration time of 1s is assumed. Do not use this as a safe-force guarantee.
    """
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('dt must be finite and positive')
    excess = np.maximum(np.asarray(strain)/.05-1, 0) + np.maximum(np.asarray(stress)/.26e6-1, 0)
    return np.minimum(1., np.asarray(damage) + dt*excess)
