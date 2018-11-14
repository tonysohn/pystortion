"""Tests for the crossmatching functions

Authors
-------

    Johannes Sahlmann

Usage
-----

    pytest -s test_crossmatch.py



"""


import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
import pytest

from ..crossmatch import xmatch

@pytest.fixture
def generate_catalogs():
    n_stars = 20
    ra = np.linspace(10, 350, n_stars)
    dec = np.linspace(-80, 80, n_stars)

    # create a n_stars x n_stars grid of stars
    ra_deg_mesh, dec_deg_mesh = np.meshgrid(ra, dec)
    ra_deg = ra_deg_mesh.flatten()
    dec_deg = dec_deg_mesh.flatten()

    primary_catalog = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
    secondary_catalog = SkyCoord(ra=ra_deg*u.degree+1.*u.arcsecond, dec=dec_deg*u.degree-1.*u.arcsecond)

    return primary_catalog, secondary_catalog


def test_crossmatch(generate_catalogs):
    primary_catalog, secondary_catalog = generate_catalogs
    xmatch_radius = 2. * u.arcsecond
    rejection_level_sigma = 3.
    verbose = True
    verbose_figures = False
    index_primary_catalog, index_secondary_catalog, d2d, d3d, delta_ra_cosdelta, delta_dec = \
        xmatch(primary_catalog, secondary_catalog, xmatch_radius, rejection_level_sigma,
                          verbose_figures=verbose_figures, verbose=verbose)

    print('Number of stars in primary_catalog   {}'.format(len(primary_catalog)))
    print('Number of stars in secondary_catalog {}'.format(len(secondary_catalog)))
    print('Number of crossmatched stars         {}'.format(len(index_primary_catalog)))

    # expect that all stars are crossmatched
    assert len(index_primary_catalog) == len(primary_catalog)

# for debugging
if __name__ == '__main__':
    test_crossmatch(generate_catalogs)
