"""Collidable segmented stalk for native MuJoCo benches.

He2024 mean geometry and tissue modulus; linear beam reduction, assumed damping
and contact friction. Abscission is a separate removable point connection.
"""
import xml.etree.ElementTree as ET
import numpy as np
from .kiwi_material import STEM_LENGTH, STEM_DIAMETER, STEM_YOUNG, STEM_AREA, STEM_I


def add_stem(root, fruit_body, fruit_anchor, junction, segments=4):
    """Attach a massive collidable stalk to the world and fruit surface.

    fruit_anchor is local to fruit_body. No contact exclusions hide the stalk
    from the fruit or hand. A 0.4 mm tip clearance avoids initial flex overlap.
    """
    length = STEM_LENGTH / segments
    radius = STEM_DIAMETER / 2
    mass = 867.5 * STEM_AREA * length
    shear = STEM_YOUNG / (2 * (1 + .26))
    parent = root.find('worldbody')
    for i in range(segments):
        pos = np.asarray(junction) + [0, 0, STEM_LENGTH] if i == 0 else [0, 0, -length]
        body = ET.SubElement(parent, 'body', name=f'stem_{i}', pos=' '.join(map(str, pos)))
        for axis, stiffness in [('1 0 0', STEM_YOUNG*STEM_I/length),
                                ('0 1 0', STEM_YOUNG*STEM_I/length),
                                ('0 0 1', shear*2*STEM_I/length)]:
            # Clamp to the first segment centre spans half a segment.
            if i == 0 and axis != '0 0 1': stiffness *= 2
            ET.SubElement(body, 'joint', type='hinge', axis=axis,
                          stiffness=str(stiffness), damping=str(.1*np.sqrt(stiffness*mass*length**2/3)))
        stiffness = STEM_YOUNG*STEM_AREA/length
        ET.SubElement(body, 'joint', type='slide', axis='0 0 -1',
                      stiffness=str(stiffness), damping=str(.1*np.sqrt(stiffness*mass)))
        clearance = .0004 if i == segments-1 else 0.
        ET.SubElement(body, 'geom', name=f'stem_collision_{i}', type='capsule',
                      fromto=f'0 0 {-radius} 0 0 {-length+radius+clearance}',
                      size=str(radius), mass=str(mass), rgba='.32 .48 .12 1',
                      contype='1', conaffinity='1', friction='.44 .005 .0001')
        parent = body
    ET.SubElement(parent, 'site', name='stem_tip', pos=f'0 0 {-length}', size='.0004', rgba='0 0 0 0')
    equality = root.find('equality')
    if equality is None: equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality, 'connect', name='abscission', body1=fruit_body,
                  body2=f'stem_{segments-1}', anchor=' '.join(map(str, fruit_anchor)),
                  solref='.0002 1', solimp='.99 .999 .0001')
