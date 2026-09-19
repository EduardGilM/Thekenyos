"""Geometry/mass consistency for the homogeneous native flesh benches only."""
from functools import lru_cache
import numpy as np

# Zhu2024 impact tissue table: Xuxiang flesh, not whole-fruit density.
FLESH_DENSITY_KG_M3 = 1030.
RADII_M = (.027, .027, .036)
FLEX_RADIUS_M = .0003
CONTACT_TIME_S = .0002  # Numerical contact response, not tissue compliance.


@lru_cache(maxsize=8)
def flesh_mass(count=7):
    """Mass from the actual tetrahedral volume, shared by matched rigid/flex tests.

    The matched rigid ellipsoid has a slightly larger volume than the faceted
    mesh. Keep their total masses equal; record that geometric approximation.
    """
    import mujoco
    if not isinstance(count, int) or count < 4:
        raise ValueError('Mesh count must be an integer >= 4')
    spacing = ' '.join(str(2*r/(count-1)) for r in RADII_M)
    xml = f'<mujoco><worldbody><flexcomp name="kiwi" type="ellipsoid" dim="3" count="{count} {count} {count}" spacing="{spacing}" radius=".0003" mass="1"/></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    tetra=d.flexvert_xpos[m.flex_elem.reshape(-1,4)]
    volume=float(np.abs(np.linalg.det(tetra[:,1:]-tetra[:,:1])/6).sum())
    return FLESH_DENSITY_KG_M3*volume
