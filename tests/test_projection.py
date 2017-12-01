from __future__ import print_function
import unittest
import numpy as np
from astropy import units as u
from ..pystortion import projection


class ProjectionTestCase(unittest.TestCase):
    def setUp(self):
    
        # center of simulated field in RA and Dec
        RA0_deg = 80.
        DE0_deg = -70.
        cosdec = np.cos(np.deg2rad(DE0_deg))

        # coordinates on grid      
        Ngrid = 10
        fieldsize = 2.*u.arcmin
        ra_max_deg = fieldsize.to(u.deg).value/2. / cosdec
        de_max_deg = fieldsize.to(u.deg).value/2.

        ra_deg = np.linspace(RA0_deg-ra_max_deg, RA0_deg+ra_max_deg, Ngrid)
        de_deg = np.linspace(DE0_deg-de_max_deg, DE0_deg+de_max_deg, Ngrid)
        ra_deg_mesh, de_deg_mesh = np.meshgrid(ra_deg, de_deg)
        ra_deg_all = ra_deg_mesh.flatten()
        de_deg_all = de_deg_mesh.flatten()

        self.ra  = ra_deg_all
        self.dec = de_deg_all
        
        # projection reference point
        self.ra_ref = RA0_deg
        self.dec_ref = DE0_deg

    def test_projection_TAN(self):
        """
        Transform from RA/Dec to tangent plane and back. check that input coordinates are recovered

        :return:
        """

        # project to detector pixel coordinates
        scale = 1.
        x,y = projection.RADec2Pix_TAN(self.ra,self.dec, self.ra_ref, self.dec_ref, scale)
        
        ra,dec = projection.Pix2RADec_TAN(x, y, self.ra_ref, self.dec_ref,scale)

        difference_modulus = np.sqrt( (self.ra-ra)**2 + (self.dec-dec)**2 )
        
        self.assertTrue(np.std(difference_modulus) < 1.e-13, 'Problem in RADec2Pix_TAN or Pix2RADec_TAN')

if __name__ == '__main__':
    unittest.main()
    
    
    
    
    