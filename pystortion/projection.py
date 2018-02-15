"""Functions for projecting spherical coordinates

Authors
-------

    Johannes Sahlmann

Use
---

"""


from astropy.modeling import models as astmodels
from astropy.modeling import rotations as astrotations

def RADec2Pix_TAN(RA, Dec, RA_ref, Dec_ref, scale=1.):
    """
    Convert RA/Dec coordinates into pixel coordinates using a tangent plane projection. The projection's reference point has to be specified.
    Scale is a convenience parameter that defaults to 1.0, in which case the returned pixel coordinates are also in degree. Scale can be set to a pixel scale to return detector coordinates in pixels

    Parameters
    ----------
    RA : float
        Right Ascension in decimal degrees
    Dec: float
        Declination in decimal degrees
    RA_ref : float
        Right Ascension of reference point in decimal degrees
    Dec_ref: float
        Declination of reference point in decimal degrees
    scale : float
        Multiplicative factor that is applied to the returned values. Default is 1.0

    Returns
    -------
	x,y : float
	   pixel coordinates in decimal degrees if scale = 1.0

    """

    # for zenithal projections, i.e. gnomonic, i.e. TAN:
    lonpole = 180.

    # tangent plane projection from phi/theta to x,y
    tan = astmodels.Sky2Pix_TAN()

    # compute native coordinate rotation to obtain phi and theta
    rot_for_tan = astrotations.RotateCelestial2Native(RA_ref, Dec_ref, lonpole)

    phi_theta = rot_for_tan(RA, Dec)

    # pixel coordinates,  x and y are in degree-equivalent
    x, y = tan(phi_theta[0], phi_theta[1])

    x = x * scale
    y = y * scale

    return x, y


def Pix2RADec_TAN(x, y, RA_ref, Dec_ref, scale=1.):
    """
    Convert pixel coordinates into RA/Dec coordinates using a tangent plane de-projection. The projection's reference point has to be specified.
    See the inverse transformation RADec2Pix_TAN.

    Parameters
    ----------
    x : float
        Pixel coordinate (default is in decimal degrees, but depends on value of scale parameter) x/scale has to be degrees.
    y : float
        Pixel coordinate (default is in decimal degrees, but depends on value of scale parameter) x/scale has to be degrees.
    RA_ref : float
        Right Ascension of reference point in decimal degrees
    Dec_ref: float
        Declination of reference point in decimal degrees
    scale : float
        Multiplicative factor that is applied to the input values. Default is 1.0

	Returns
    -------
    RA : float
        Right Ascension in decimal degrees
    Dec: float
        Declination in decimal degrees

    """
    # for zenithal projections, i.e. gnomonic, i.e. TAN
    lonpole = 180.

    x = x / scale
    y = y / scale

    # tangent plane projection from x,y to phi/theta
    tan = astmodels.Pix2Sky_TAN()

    # compute native coordinate rotation to obtain RA and Dec
    rot_for_tan = astrotations.RotateNative2Celestial(RA_ref, Dec_ref, lonpole)

    phi, theta = tan(x, y)

    # RA and Dec
    Ra, Dec = rot_for_tan(phi, theta)

    return Ra, Dec
