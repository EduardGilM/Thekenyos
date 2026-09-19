"""Reduced-stalk geometry and small-deflection beam checks."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from treesim.native_stem import add_stem
from treesim.kiwi_material import STEM_LENGTH, STEM_AREA, STEM_YOUNG, STEM_I

try:
    import mujoco
except ImportError:
    mujoco = None


class StemTests(unittest.TestCase):
    def scene(self):
        root=ET.fromstring('<mujoco><option gravity="0 0 0" timestep=".000005" integrator="implicitfast"/><worldbody><body name="fruit"><freejoint/><geom type="sphere" size=".025" mass=".1"/></body></worldbody></mujoco>')
        add_stem(root,'fruit',[0,0,.025],[0,0,.025])
        return root

    def test_beam_compliance(self):
        root=self.scene();bodies=[root.find(f'.//body[@name="stem_{i}"]') for i in range(4)]
        axial=sum(1/float(b.find('joint[@type="slide"]').get('stiffness')) for b in bodies)
        self.assertAlmostEqual(axial,STEM_LENGTH/(STEM_YOUNG*STEM_AREA))
        bending=sum((STEM_LENGTH*(1-i/4))**2/float(b.find('joint[@axis="1 0 0"]').get('stiffness')) for i,b in enumerate(bodies))
        continuum=STEM_LENGTH**3/(3*STEM_YOUNG*STEM_I)
        self.assertLess(abs(bending/continuum-1),.04)
        for body in bodies:
            geom=body.find('geom');self.assertEqual(geom.get('contype'),'1');self.assertEqual(geom.get('conaffinity'),'1')
        self.assertEqual(root.find('equality/connect').get('body1'),'fruit')

    @unittest.skipIf(mujoco is None,'MuJoCo environment required')
    def test_physical_tip_deflection(self):
        m=mujoco.MjModel.from_xml_string(ET.tostring(self.scene(),encoding='unicode'));d=mujoco.MjData(m)
        d.eq_active[:]=False;mujoco.mj_forward(m,d)
        site=m.site('stem_tip').id;body=m.body('stem_3').id;origin=d.site_xpos[site].copy()
        for _ in range(400000):
            d.qfrc_applied[:]=0
            mujoco.mj_applyFT(m,d,np.array([.001,0,0]),np.zeros(3),d.site_xpos[site],body,d.qfrc_applied)
            mujoco.mj_step(m,d)
        expected=.001*STEM_LENGTH**3/(3*STEM_YOUNG*STEM_I)
        measured=d.site_xpos[site,0]-origin[0]
        self.assertLess(abs(measured/expected-1),.04)
        self.assertFalse(any(w.number for w in d.warning))
        self.assertAlmostEqual(sum(m.body_mass[m.body(f'stem_{i}').id] for i in range(4)),867.5*STEM_AREA*STEM_LENGTH)


if __name__=='__main__':unittest.main()
