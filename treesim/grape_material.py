"""SI table-grape cluster presets. Provenance and limits: docs/grape-material-evidence.md.

Measured envelopes come from Hu, Zhang & Hu 2025 (Kyoho/Rose/Red Globe). Peduncle
elasticity, tensile detachment, peduncle length and berry size are explicit
assumptions or proxies, never measured cultivar values.
"""
import numpy as np

# Hu2025 Table 1, min-max over three varieties (measured).
CLUSTER_LENGTH_RANGE = (.130, .290)        # m
CLUSTER_DIAMETER_RANGE = (.090, .200)      # m, equatorial
CLUSTER_MASS_RANGE = (.300, 2.000)         # kg
PEDUNCLE_DIAMETER_RANGE = (.0046, .0108)   # m
# Hu2025 Table 2 (measured). Single-berry plate tests / blade shear, not whole-cluster limits.
BERRY_RUPTURE_FORCE_MIN = 25.79            # N, longitudinal compression, Rose
PEDUNCLE_SHEAR_FORCE_RANGE = (85.37, 92.36)  # N, blade cutting load
# Engineering assumptions (see docs section 3/4).
BULK_DENSITY_RANGE = (250., 700.)          # kg/m^3, ellipsoid bulk density acceptance
PEDUNCLE_YOUNG = 300e6                     # Pa, assumed woody-stem order of magnitude
PEDUNCLE_LENGTH_RANGE = (.040, .060)       # m, assumed
DETACH_RANGE = (40., 90.)                  # N, assumed tensile pull-off proxy (<= measured shear)
BERRY_RADIUS_RANGE = (.010, .013)          # m, visual only
# Contact proxies reused from the kiwi bench; not grape measurements.
FRUIT_FRICTION = .6
RESTITUTION = .05


def sample_cluster(rng):
    """Rejection sample Hu2025 length/diameter/mass under the assumed bulk-density bound.

    Returns (radii xyz semi-axes [m], mass [kg], peduncle_diameter [m], peduncle_length [m]).
    """
    for _ in range(10000):
        length = rng.uniform(*CLUSTER_LENGTH_RANGE)
        diameter = rng.uniform(*CLUSTER_DIAMETER_RANGE)
        mass = rng.uniform(*CLUSTER_MASS_RANGE)
        radii = np.array([diameter/2, diameter/2, length/2])
        density = mass/(4*np.pi*np.prod(radii)/3)
        if BULK_DENSITY_RANGE[0] <= density <= BULK_DENSITY_RANGE[1]:
            return radii, float(mass), float(rng.uniform(*PEDUNCLE_DIAMETER_RANGE)), \
                float(rng.uniform(*PEDUNCLE_LENGTH_RANGE))
    raise RuntimeError('grape cluster geometry/mass rejection sampler failed')


def peduncle_stiffness(diameter, length):
    """Axial [N/m], bending [N/m] and rotational [N m/rad] beam stiffness (assumed E)."""
    area = np.pi*diameter**2/4
    inertia = np.pi*diameter**4/64
    return (PEDUNCLE_YOUNG*area/length, 3*PEDUNCLE_YOUNG*inertia/length**3,
            PEDUNCLE_YOUNG*inertia/length)
