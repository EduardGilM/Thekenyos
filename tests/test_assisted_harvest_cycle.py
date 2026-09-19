"""Containment accepts physical wall contact but rejects a protruding fruit."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.assisted_harvest_cycle import inside_basket
from treesim.basket import CENTER, SIZE, WALL


class BasketContainmentTest(unittest.TestCase):
    def test_resting_against_wall_and_floor(self):
        for side in (-1, 1):
            center=CENTER+[side*(SIZE[0]/2-WALL-.027),0,WALL/2+.036]
            self.assertTrue(inside_basket(center,np.eye(3),np.zeros(3)))
            self.assertTrue(inside_basket(center+[side*1e-8,0,0],np.eye(3),np.zeros(3)))
            self.assertFalse(inside_basket(center+[side*.0001,0,0],np.eye(3),np.zeros(3)))

    def test_rotated_fruit_and_translated_basket(self):
        base=np.array([1.,-2.,.65]);rotation=Rotation.from_euler('y',90,degrees=True).as_matrix()
        center=base+CENTER+[SIZE[0]/2-WALL-.036,0,WALL/2+.027]
        self.assertTrue(inside_basket(center,rotation,base))
        self.assertFalse(inside_basket(center+[.0001,0,0],rotation,base))

    def test_above_rim_and_below_floor(self):
        center=CENTER+[0,0,WALL/2+.036]
        self.assertFalse(inside_basket(center+[0,0,-.0001],np.eye(3),np.zeros(3)))
        self.assertFalse(inside_basket(CENTER+[0,0,SIZE[2]],np.eye(3),np.zeros(3)))


if __name__=='__main__':
    unittest.main()
