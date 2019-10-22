"""Classes and functions to support analysis and interpretation of geometric distortion in astronomical imaging
instruments

Authors
-------

    Johannes Sahlmann

Use
---
    import distortion



"""


from __future__ import print_function
import os
import sys
import copy
import numpy as np
import pylab as pl
from matplotlib.ticker import FormatStrFormatter
from astropy import units as u
from astropy.table import Table, Column
from astropy.table import hstack as tablehstack

from astropy.coordinates import SkyCoord
import scipy

import sympy
from sympy.abc import x, y
from sympy import MatrixSymbol, Matrix
from sympy.utilities.lambdify import lambdify
from sympy.abc import x as sympy_x
from sympy.abc import y as sympy_y

import scipy.spatial
import matplotlib

from linearfit import linearfit
from uhelpers import plotting_helpers

try:
    from kapteyn import kmpfit
except ImportError:
    print('kapteyn package is not available')
    pass

from .utils import plot_spatial_difference


deg2mas = u.deg.to(u.mas)

def moving_average(a, n=3):
    """
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy/14314054
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # return ret[n - 1:] / n
    return ret / n



class multiEpochAstrometry(object):
    """Class for multi-epoch astrometry of matched sources"""

    def __init__(self, p, col_names, aux=None):
        self.p = p
        self.colNames = col_names
        self.aux = aux # astropy table
        self.add_usable_flag()
        if 'id' in self.colNames:
            self.star_id_index = self.colNames.tolist().index('id')

    def add_usable_flag(self, mode='default'):
        """Add weight parameters to the `p` array.

        Parameters
        ----------
        mode

        Returns
        -------

        """
        if mode == 'default':
            usable_flag = np.ones(self.p.shape[:2])
            self.p = np.dstack((self.p, usable_flag))
            self.colNames = np.hstack((self.colNames, 'usable_flag'))

    def set_target_first(self, target_number):
        """
        reconstruct p-array from good reference stars and target
        target is first element

        :param target_number:
        :return:
        """

        s = self.reference_star_data(target_number)
        self.p = np.concatenate((self.p[:, np.where((self.p[0, :, self.star_id_index] == target_number))[0], :], s), axis=1)
        target_index = 0
        return target_index

    def reference_star_data(self, target_number):
        """
        remove target from p data array

        :param p:
        :param target_number:
        :return:
        """
        s = self.p[:, np.where((self.p[0, :, self.star_id_index] != target_number))[0], :]
        return s

    def remove_frames(self, frames_to_remove):
        """Remove frames that correspond to the index in self.p given by frames_to_remove.
        """
        a = np.arange(len(self.p))
        good_idx = np.setdiff1d(a, frames_to_remove)
        self.p = self.p[good_idx, :, :]
        self.aux = self.aux[good_idx]

    def remove_star(self, star_id):
        """Remove a star identified by its id from the data array.

        Parameters
        ----------
        star_id : float or int

        """
        self.p = np.delete(self.p, np.where((self.p[0, :, self.star_id_index] == star_id))[0], 1)




def prepare_multi_epoch_astrometry(star_catalog_matched, reference_catalog_matched, fieldname_dict=None):
    """Return a multiEpochAstrometry object with two catalogs that are populated from the arguments.

    Parameters
    ----------
    star_catalog_matched : astropy table
        Table with one catalog
    reference_catalog_matched : astropy table
        Table with second catalog with matched rows for first catalog
    fieldname_dict : dict
        Specifies the table columns to be used

    Returns
    -------
    mp : multiEpochAstrometry instance
        Standardised format for matched astrometry

    """
    if fieldname_dict is None:
        # dictionary to allow for flexible field/column names
        fieldname_dict = {}
        fieldname_dict['star_catalog'] = {}
        fieldname_dict['reference_catalog'] = {}

        fieldname_dict['reference_catalog']['position_1'] = 'v2_tangent_arcsec'
        fieldname_dict['reference_catalog']['position_2'] = 'v3_tangent_arcsec'
        fieldname_dict['reference_catalog']['sigma_position_1'] = 'v2_error_arcsec'
        fieldname_dict['reference_catalog']['sigma_position_2'] = 'v3_error_arcsec'
        fieldname_dict['reference_catalog']['identifier'] = 'source_id'
        fieldname_dict['reference_catalog']['position_unit'] = u.arcsecond
        fieldname_dict['reference_catalog']['sigma_position_unit'] = u.arcsecond

        fieldname_dict['star_catalog']['position_1'] = 'v2_tangent_arcsec'
        fieldname_dict['star_catalog']['position_2'] = 'v3_tangent_arcsec'
        fieldname_dict['star_catalog']['sigma_position_1'] = 'sigma_x_arcsec'
        fieldname_dict['star_catalog']['sigma_position_2'] = 'sigma_y_arcsec'
        fieldname_dict['star_catalog']['identifier'] = 'star_id'
        fieldname_dict['star_catalog']['position_unit'] = u.arcsecond
        fieldname_dict['star_catalog']['sigma_position_unit'] = u.arcsecond

    # number of crossmatched stars
    Nstars = len(star_catalog_matched)
    colNames = np.array(['x', 'y', 'sigma_x', 'sigma_y', 'id', 'original_id'])
    p = np.zeros((2, Nstars, len(colNames)))
    mp = multiEpochAstrometry(p, colNames)

    i_x = np.where(mp.colNames == 'x')[0][0]
    i_y = np.where(mp.colNames == 'y')[0][0]
    i_sx = np.where(mp.colNames == 'sigma_x')[0][0]
    i_sy = np.where(mp.colNames == 'sigma_y')[0][0]
    i_id = np.where(mp.colNames == 'id')[0][0]
    i_oid = np.where(mp.colNames == 'original_id')[0][0]

    gnames = np.array(reference_catalog_matched[fieldname_dict['reference_catalog']['identifier']])
    hnames = np.array(star_catalog_matched[fieldname_dict['star_catalog']['identifier']]).astype(np.int)
    xmatchId = gnames


    for index in [0,1]:
        if index == 0: # first catalog (instrument)
            catalog = copy.deepcopy(star_catalog_matched)
            catalog_name = 'star_catalog'
            star_names = np.array(catalog[fieldname_dict[catalog_name]['identifier']])
            xmatchId = star_names.copy()
        elif index == 1: # second catalog (reference, e.g. Gaia)
            catalog = copy.deepcopy(reference_catalog_matched)
            catalog_name = 'reference_catalog'
            star_names = np.array(catalog[fieldname_dict[catalog_name]['identifier']]).astype(np.int)

        mp.p[index, :, [i_x, i_y, i_sx, i_sy, i_id, i_oid]] = np.vstack((
            np.array(catalog[fieldname_dict[catalog_name]['position_1']]),
            np.array(catalog[fieldname_dict[catalog_name]['position_2']]),
            np.array(catalog[fieldname_dict[catalog_name]['sigma_position_1']]) * fieldname_dict[catalog_name]['sigma_position_unit'].to(fieldname_dict[catalog_name]['position_unit']),
            np.array(catalog[fieldname_dict[catalog_name]['sigma_position_2']]) * fieldname_dict[catalog_name]['sigma_position_unit'].to(fieldname_dict[catalog_name]['position_unit']),
            xmatchId, star_names))

    return mp


def stdout_header(str):
    print('=' * 30, ' %s ' % str, '=' * 30)


def stdout_section(str):
    print('-' * 20, ' %s ' % str, '-' * 20)


class lazAstrometryCoefficients(object):
    """Class for Lazorenko-style astrometry"""

    def __init__(self, k=None, p=None, p_dif=9999, p_red=9999, Alm=9999, rms=9999, C=9999, degree=9999, maxdeg=9999,
                 Nalm=None, resx=None, resy=None, s_Alm_normal=None, s_Alm_formal=None, s_p_red=None,
                 polynomialTermOrder=None, referencePoint=None, useReducedCoordinates=None, refFrameNumber=None,
                 colNames=None, data=None):

        self.p = p
        self.p_dif = p_dif
        self.p_red = p_red
        self.k = k
        self.Alm = Alm
        self.rms = rms
        self.C = C
        self.degree = degree
        self.maxdeg = maxdeg
        self.Nalm = Nalm
        self.resx = resx
        self.resy = resy
        self.s_Alm_formal = s_Alm_formal
        self.s_Alm_normal = s_Alm_normal
        self.s_p_red = s_p_red
        self.polynomialTermOrder = polynomialTermOrder  # array of strings, e.g. [1, x, y, x**2, x*y, y**2, x**3, x**2*y,...]
        self.referencePoint = referencePoint
        self.useReducedCoordinates = useReducedCoordinates
        self.refFrameNumber = refFrameNumber
        self.colNames = colNames
        self.data=data

    def display_results(self, evaluation_frame_number=1, precision=7, scale_factor_for_residuals=1., display_correlations=False,
                        nformat='f', print_rms_only=False):

        scaleFactor = scale_factor_for_residuals
        names = ['X', 'Y']
        resFields = ['resx', 'resy']

        if print_rms_only:
            print('RMS  in X and Y: {0:f}  and  {1:f}'.format(
                self.rms[evaluation_frame_number][0] * scale_factor_for_residuals,
                self.rms[evaluation_frame_number][1] * scale_factor_for_residuals))
            return

        for i, r in enumerate(resFields):
            print('Polynomial parameters in {0:s}:'.format(names[i]))
            eval('self.{0:s}[{1:d}]'.format(r, evaluation_frame_number)).display_results(precision=precision,
                                                                                 scale_factor=scaleFactor,
                                                                                 nformat=nformat)
            if display_correlations:
                eval('self.{0:s}[{1:d}]'.format(r, evaluation_frame_number)).display_correlations()

        print('RMS  in X and Y: {0:f}  and  {1:f}'.format(self.rms[evaluation_frame_number][0] * scale_factor_for_residuals,
                                                          self.rms[evaluation_frame_number][1] * scale_factor_for_residuals))
        print('        chi2 in X and Y: {0:f}  and  {1:f}'.format(self.resx[evaluation_frame_number].chi2,
                                                                  self.resy[evaluation_frame_number].chi2))
        print('reduced chi2 in X and Y: {0:f}  and  {1:f}'.format(
            self.resx[evaluation_frame_number].chi2 / self.resx[evaluation_frame_number].n_freedom,
            self.resy[evaluation_frame_number].chi2 / self.resy[evaluation_frame_number].n_freedom))

        if (self.k >= 4):
            # display classical quantities: shift, rotation, skew (only meaningful if reduced coordinates are not used)
            self.human_readable_solution_parameters = displayRotScaleSkew(self, i=evaluation_frame_number,
                                                                          scaleFactor=scaleFactor, nformat=nformat)

    def plotResiduals(self, evaluation_frame_number, outDir, nameSeed, omc_scale=1., save_plot=1, omc_unit='undefined',
                      xy_unit='undefined', xy_scale=1., title=None, verbose=False, target_number=None, plot_correlations=False,
                      plot_apertures=None, **kwargs):
        """

        2018-08-13 add target_index argument to exlude target residuals from plotting

        Parameters
        ----------
        evaluation_frame_number
        outDir
        nameSeed
        omc_scale
        save_plot
        omc_unit
        xy_unit
        xy_scale
        title
        verbose
        target_index

        Returns
        -------

        """


        ii = evaluation_frame_number
        ix = np.where(self.colNames == 'x')[0][0]
        iy = np.where(self.colNames == 'y')[0][0]
        i_id = np.where(self.colNames == 'id')[0][0]

        # plot only non-artifical measurements as those are used to constrain the fit
        plot_index = np.arange(len(self.resx[ii].residuals))
        if 'artificial' in self.colNames:
            i_artificial = np.where(self.colNames == 'artificial')[0][0]
            plot_index = np.where(self.p[ii, :, i_artificial] == 0)[0]

        # reomve target from plotting index
        if target_number is not None:
            target_number_index = list(self.p[0, :, i_id]).index(target_number)
            if (target_number_index is not None) and (target_number_index in plot_index):
                plot_index = np.delete(plot_index, np.where(plot_index==target_number_index)[0])

        xlabl = 'X (%s)' % xy_unit
        ylabl = 'Y (%s)' % xy_unit

        U = self.resx[ii].residuals[plot_index] * omc_scale
        V = self.resy[ii].residuals[plot_index] * omc_scale

        if verbose:
            print('maximum residual amplitude %3.3f ' % (np.max(np.sqrt(U ** 2 + V ** 2))))

        omc = np.vstack((U, V)).T
        plotting_helpers.histogram_with_gaussian_fit(omc, facecolors=['b', 'r'], linecolors=['b', 'r'], labels=['X', 'Y'],
                                    xlabel='Residual O-C (%s)' % omc_unit, normed=1, save_plot=save_plot, out_dir=outDir,
                                    name_seed=nameSeed, separate_panels=True, titles=title, **kwargs)

        # plot residuals on sky
        fig = pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
        pl.clf()
        # xy = np.ma.masked_array(self.p[ii, :, [ix, iy]], mask=[self.p[ii, :, [ix, iy]] == 0]) * xy_scale
        xy = np.ma.masked_array(self.p[ii, plot_index, np.array([ix, iy])[:, np.newaxis]], mask=[self.p[ii, plot_index, np.array([ix, iy])[:, np.newaxis]] == 0]) * xy_scale
        # xy = self.p[ii, plot_index, [ix, iy]] * xy_scale
        # xy = self.p[ii, plot_index, [ix, iy]] * xy_scale

        pl.quiver(xy[0], xy[1], U, V, angles='xy')
        pl.axis('tight')
        pl.axis('equal')
        pl.xlabel('X (%s)' % xy_unit)
        pl.ylabel('Y (%s)' % xy_unit)
        title_string = 'Residuals (k={})'.format(self.k)
        if title is not None:
            title_string = '{}, {}'.format(title, title_string)
        pl.title(title_string)
        if plot_apertures is not None:
            ax = pl.gca()
            for aperture in plot_apertures:
                aperture.plot(ax=ax, fill_color='none', color='0.7', label=True)

        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_distortionResidualsOnSky.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)


        cm = pl.cm.get_cmap('RdYlBu')
        #         cm = pl.cm.get_cmap('Greys')
        axes = ['X', 'Y']
        fig = pl.figure(figsize=(12, 5), facecolor='w', edgecolor='k')
        pl.clf()
        for j, axis in enumerate(axes):
            pl.subplot(1, 2, j + 1)
            if axis == 'X':
                z = U
            elif axis == 'Y':
                z = V
            if 1 == 0:
                maxResidual = 3.
                sc = pl.scatter(xy[0], xy[1], c=z, cmap=cm, edgecolors='face', alpha=0.8, vmin=-maxResidual,
                                vmax=maxResidual, s=10)  # , vmin=0, vmax=20, s=35
            else:
                sc = pl.scatter(xy[0], xy[1], c=z, cmap=cm, edgecolors='face', alpha=0.8,
                                s=20)  # , vmin=0, vmax=20, s=35
            pl.colorbar(sc)
            pl.title('Residuals in %s' % (axis))
            pl.axis('equal')
            pl.xlabel(xlabl)
            pl.ylabel(ylabl)
            if plot_apertures is not None:
                ax = pl.gca()
                for aperture in plot_apertures:
                    aperture.plot(ax=ax, fill_color='none', color='0.7')

        fig.tight_layout(h_pad=0.0)
        if title is not None:
            pl.title(title)

        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_residuals_sky.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)



        n_moving_average = 500
        fig = pl.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
        pl.clf()
        for jj, axis_name in enumerate(['X', 'Y']):

            pl.subplot(1, 2, jj+1)
            x_plot = xy[jj]
            y_plot = getattr(self, 'res{}'.format(axis_name.lower()))[ii].residuals[plot_index] * omc_scale
            sort_index = np.argsort(x_plot)
            if jj == 0:
                pl.plot(x_plot, y_plot, 'b.')
            else:
                pl.plot(x_plot, y_plot, 'r.')
            pl.plot(x_plot[sort_index], moving_average(y_plot[sort_index], n=n_moving_average), 'k-')
            pl.xlabel('{} ({})'.format(axis_name, xy_unit))
            pl.ylabel('Residuals in {} ({})'.format(axis_name, omc_unit))
            if (jj==0) & (title is not None):
                pl.title(title)

        # pl.plot(xy[0], moving_average(self.resx[ii].residuals[plot_index] * omc_scale, n=n_moving_average), 'k-')
        # pl.xlabel('X (%s)' % xy_unit)
        # pl.ylabel('Residuals in X (%s)' % omc_unit)
        # pl.subplot(1, 2, 2)
        # pl.plot(xy[1], self.resy[ii].residuals[plot_index] * omc_scale, 'r.')
        # pl.plot(xy[1], moving_average(self.resy[ii].residuals[plot_index] * omc_scale, n=n_moving_average), 'k-')
        # pl.xlabel('Y (%s)' % xy_unit)
        # pl.ylabel('Residuals in Y (%s)' % omc_unit)
        fig.tight_layout(h_pad=0.0)
        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_distortionResidualsVsRADec.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)



        if plot_correlations:
            fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
            pl.clf()
            pl.plot(U, V, 'bo', label='X')
            pl.title('Distortion Residuals X vs Y')
            pl.ylabel('Residual O-C Y ({})'.format(omc_unit))
            pl.xlabel('Residual O-C X ({})'.format(omc_unit))
            pl.show()

            for name in self.colNames:
                # if name in 'x y CHIP_EXTENSION'.split():
                if name not in 'MAG sigma_x sigma_y'.split():
                    continue
            #         'MAG', 'FLAGS_EXTRACTION', 'x', 'y', 'index', 'sigma_world', 'id',
            # 'CHIP_EXTENSION', 'sigma_x', 'sigma_y', 'CATALOG_NUMBER',
            # 'artificial', 'usable_flag'
                p_index = np.where(self.colNames == name)[0][0]
                fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
                pl.clf()
                pl.plot(self.p[ii, plot_index, p_index], U, 'b.', label='X')
                pl.plot(self.p[ii, plot_index, p_index], V, 'r.', label='Y')
                pl.title('Distortion Residuals vs {}'.format(name))
                pl.xlabel(name)
                pl.ylabel('Residual O-C ({})'.format(omc_unit))
                pl.show()



    def plotResults(self, evaluation_frame_number, outDir, nameSeed, saveplot=1, xy_unit='undefined', xy_scale=1.):
        """

        Parameters
        ----------
        evaluation_frame_number
        outDir
        nameSeed
        saveplot
        xy_unit
        xy_scale

        Returns
        -------

        """
        Nframes, Nstars, Nfields = self.p.shape
        Nalm = self.Nalm

        ix = np.where(self.colNames == 'x')[0][0]
        iy = np.where(self.colNames == 'y')[0][0]

        U = np.zeros((Nframes, Nstars))
        V = np.zeros((Nframes, Nstars))
        for i in range(Nstars):  # loop over reference stars
            #         for reconstruction of PHI from the A_lm parameters previously measured on reference stars in each frame
            PHIx = self.C[:, i].T * self.Alm[:, 0:Nalm].T  # for each frame this gives PHIx of Thesis Eq. 4.19
            PHIy = self.C[:, i].T * self.Alm[:, Nalm:].T
            U[:, i] = np.array(PHIx)[0]
            V[:, i] = np.array(PHIy)[0]

        if self.useReducedCoordinates == 0:
            #       PHIx/PHIy correspond to corrections to the reduced coordinates in Thesis Eq. 4.19, if reduced coordinates are not used, we need to subtract the positions in the reference frame, which are fitted in the model
            for j in range(Nframes):
                U[j, :] = U[j, :] - self.p[self.refFrameNumber, :, ix]
                V[j, :] = V[j, :] - self.p[self.refFrameNumber, :, iy]

        mU = np.subtract(U.T, np.mean(U, axis=1)).T
        mV = np.subtract(V.T, np.mean(V, axis=1)).T

        ii = evaluation_frame_number

        # display modelled distortion
        xlabl = 'X (%s)' % xy_unit
        ylabl = 'Y (%s)' % xy_unit
        xy = np.ma.masked_array(self.p[ii, :, [ix, iy]], mask=[self.p[ii, :, [ix, iy]] == 0]) * xy_scale

        fig = pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k')
        pl.clf()
        pl.subplot(1, 2, 1)
        pl.quiver(xy[0], xy[1], U[ii, :], V[ii, :], angles='xy')
        #         pl.streamplot(xy[0],xy[1], U[ii,:], V[ii,:])

        pl.axis('equal')
        pl.xlabel(xlabl)
        pl.ylabel(ylabl)
        pl.title('Distortion model (k=%d)' % self.k)
        pl.subplot(1, 2, 2)
        pl.quiver(xy[0], xy[1], mU[ii, :], mV[ii, :], angles='xy')
        #         pl.streamplot(xy[0],xy[1], mU[ii,:], mV[ii,:])
        pl.axis('equal')
        pl.xlabel(xlabl)
        pl.ylabel(ylabl)
        pl.title('Distortion model (k=%d), shift subtracted' % self.k)
        fig.tight_layout(h_pad=0.0)
        pl.show()
        if saveplot == 1:
            figName = os.path.join(outDir, '%s_distortionModel.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    def getHullPath(self, points):
        hull = scipy.spatial.ConvexHull(points)
        path = matplotlib.path.Path(points[hull.vertices,])
        return path


    def _set_footprint_grid(self, n_grid=100, remove_masked=True, evaluation_frame_number=1):
        """Set x,y grid coordinates that sample the footprint."""

        polynomial_evaluation_grid = copy.deepcopy(self.C) # ximinusx0

        # add back reference point
        polynomial_evaluation_grid[1,:] += 1*self.referencePoint[evaluation_frame_number, 0]
        polynomial_evaluation_grid[2,:] += 1*self.referencePoint[evaluation_frame_number, 1]

        points = np.array(polynomial_evaluation_grid[[1, 2],], dtype='float32').T
        path = self.getHullPath(points)

        x_min = np.min(polynomial_evaluation_grid[1,])
        x_max = np.max(polynomial_evaluation_grid[1,])
        y_min = np.min(polynomial_evaluation_grid[2,])
        y_max = np.max(polynomial_evaluation_grid[2,])
        # x_min = np.min(self.C[1,])
        # x_max = np.max(self.C[1,])
        # y_min = np.min(self.C[2,])
        # y_max = np.max(self.C[2,])

        xx = np.linspace(x_min, x_max, n_grid)
        yy = np.linspace(y_min, y_max, n_grid)
        x_mesh, y_mesh = np.meshgrid(xx, yy)

        goodMeshMask = path.contains_points(np.array([x_mesh.flatten(), y_mesh.flatten()]).T).reshape(x_mesh.shape)
        x_mesh = np.ma.masked_array(x_mesh, mask=~goodMeshMask)
        y_mesh = np.ma.masked_array(y_mesh, mask=~goodMeshMask)

        if remove_masked:
            # return unmasked arrays of valid grid points
            index = np.where(x_mesh.mask == False)
            x_mesh = np.array(x_mesh.data[index])
            y_mesh = np.array(y_mesh.data[index])

        self.x_footprint = x_mesh
        self.y_footprint = y_mesh


    def plot_distortion_offsets(self, evaluation_frame_number=1, plot_dir=os.path.expanduser('~'),
                                name_seed='distortion_offsets', siaf=None, plot_aperture_names=None,
                                verbose=True):

        if hasattr(self, 'x_footprint') is False:
            self._set_footprint_grid(n_grid=10)

        # fig = pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        # pl.clf()

        data = {}
        data['reference'] = {'x': self.x_footprint, 'y': self.y_footprint}

        xx = copy.deepcopy(self.x_footprint)
        yy = copy.deepcopy(self.y_footprint)

        x_distorted, y_distorted = self.apply_polynomial_transformation(evaluation_frame_number, xx, yy)

        if verbose:
            print('*'*20)
            print('Illustration of the effect of the distortion polynomial')
            print('Reference point for this polynomial is {0[0]}, {0[1]}'.format(self.referencePoint[evaluation_frame_number]))
            x_ref_distorted, y_ref_distorted = self.apply_polynomial_transformation(evaluation_frame_number, self.referencePoint[evaluation_frame_number, 0], self.referencePoint[evaluation_frame_number, 1])
            print('Transformed reference point is at      {0}, {1}'.format(x_ref_distorted, y_ref_distorted))
            if siaf is not None:
                for aperture_name in plot_aperture_names:
                    aperture = siaf[aperture_name]
                    print('{}  V2Ref={:2.3f} V3Ref={:2.3f}'.format(aperture_name, aperture.V2Ref_original, aperture.V3Ref_original))
                    x_apref_distorted, y_apref_distorted = self.apply_polynomial_transformation(
                        evaluation_frame_number, aperture.V2Ref_original, aperture.V3Ref_original)
                    print('Transformed V2Ref={:2.3f}  V3Ref={:2.3f}'.format(x_apref_distorted, y_apref_distorted))
                    print('Difference dV2Ref={:2.3f} dV3Ref={:2.3f}'.format(x_apref_distorted-aperture.V2Ref_original, y_apref_distorted-aperture.V3Ref_original))
            print('*'*20)

        data['comparison_0'] = {'x': x_distorted, 'y': y_distorted}

        plot_spatial_difference(data,
                                                  figure_types=['data', 'quiver', 'offset-corrected-quiver'],
                                                  xy_label=['$\\nu_2$', '$\\nu_3$'], xy_unit='arcsec',
                                                  plot_dir=plot_dir,
                                                  name_seed=name_seed, siaf=siaf,
                                                  plot_aperture_names=plot_aperture_names)


    def plotRotation(self, evaluation_frame_number, outDir, nameSeed, referencePointForProjection_Pix, save_plot=1,
                     xy_unit='undefined', xy_scale=1.):
        """

        Parameters
        ----------
        evaluation_frame_number
        outDir
        nameSeed
        referencePointForProjection_Pix
        save_plot
        xy_unit
        xy_scale

        Returns
        -------

        """
        ii = evaluation_frame_number
        ix = np.where(self.colNames == 'x')[0][0]
        iy = np.where(self.colNames == 'y')[0][0]
        xlabl = 'X (%s)' % xy_unit
        ylabl = 'Y (%s)' % xy_unit

        P = sympy.symbols('p0:%d' % self.Nalm)
        Cs, polynomialTermOrder = bivariate_polynomial(x, y, self.k, verbose=0)
        #         polynomial = Matrix(Cs.T)*Matrix(P)
        partialMode = 4
        includeAllHigherOrders = False
        polynomial = self.get_polynomial(Cs, P, partial_mode=partialMode, include_all_higher_orders=includeAllHigherOrders)

        points = np.array(self.C[[1, 2],], dtype='float32').T
        #         hull = scipy.spatial.ConvexHull(points)
        #         path = matplotlib.path.Path(points[hull.vertices,])
        path = self.getHullPath(points)

        axes = ['X', 'Y']
        x_min = np.min(self.C[1,])  # ximinusx0
        x_max = np.max(self.C[1,])
        y_min = np.min(self.C[2,])
        y_max = np.max(self.C[2,])

        Ngrid = 100
        xx = np.linspace(x_min, x_max, Ngrid)
        yy = np.linspace(y_min, y_max, Ngrid)
        x_mesh, y_mesh = np.meshgrid(xx, yy)

        goodMeshMask = path.contains_points(np.array([x_mesh.flatten(), y_mesh.flatten()]).T).reshape(x_mesh.shape)
        x_mesh = np.ma.masked_array(x_mesh, mask=~goodMeshMask)
        y_mesh = np.ma.masked_array(y_mesh, mask=~goodMeshMask)

        #       b = dx/dx   c = dx/dy    e = dy/dx   f=dy/dy
        #       xrot = atan(-e,b)       yrot = atan(c,f)
        #       xscale = np.sqrt(b**2+c**2)     yscale = np.sqrt(e**2+f**2)

        #         import functools
        #         def compose2(*functions):
        #             return functools.reduce(lambda f1, f2: lambda x,y: np.rad2deg(np.arctan2(f1(x,y),f2(x,y))), functions, lambda x,y: x,y)
        #         def compose2(f1,f2):
        #             return lambda x,y: np.rad2deg(np.arctan2(f1(x,y),f2(x,y)))

        fig = pl.figure(figsize=(12, 16), facecolor='w', edgecolor='k')
        pl.clf()
        referencePointRotation = np.zeros(2)
        referencePointScale = np.zeros(2)
        for j, axis in enumerate(axes):
            replacements1 = (','.join(['("p%d",self.Alm[%d,%d])' % (d, ii, 0 + d) for d in range(self.Nalm)]))
            replacements2 = (','.join(['("p%d",self.Alm[%d,%d])' % (d, ii, self.Nalm + d) for d in range(self.Nalm)]))
            if axis == 'X':
                dfexpr = sympy.diff(polynomial, x)
                dffunc1 = dfexpr.subs(eval('[' + replacements1 + ']'))
                dffunc2 = dfexpr.subs(eval('[' + replacements2 + ']'))
                b_func = lambdify((x, y), dffunc1, 'numpy')
                e_func = lambdify((x, y), dffunc2, 'numpy')
                b = b_func(x_mesh, y_mesh)[0][0]
                e = e_func(x_mesh, y_mesh)[0][0]
                rot = np.rad2deg(np.arctan2(-e, b))
                scale = np.sqrt(e ** 2 + b ** 2)
                #                 rotfunc = np.rad2deg(np.arctan2(-e_func,b_func))
                #                 rotfunc = compose2(-e_func,b_func)
                referencePointRotation[j] = np.rad2deg(
                    np.arctan2(-e_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0],
                               b_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]))
                referencePointScale[j] = np.sqrt(
                    e_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0] ** 2 +
                    b_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0] ** 2)

            elif axis == 'Y':
                dfexpr = sympy.diff(polynomial, y)
                dffunc1 = dfexpr.subs(eval('[' + replacements1 + ']'))
                dffunc2 = dfexpr.subs(eval('[' + replacements2 + ']'))
                c_func = lambdify((x, y), dffunc1, 'numpy')
                f_func = lambdify((x, y), dffunc2, 'numpy')
                c = c_func(x_mesh, y_mesh)[0][0]
                f = f_func(x_mesh, y_mesh)[0][0]
                #                 rotfunc = compose2(c_func,f_func)
                #                 rotfunc = np.rad2deg(np.arctan2(c_func,f_func))
                #                 rot = rotfunc(x_mesh,y_mesh)[0][0]
                rot = np.rad2deg(np.arctan2(c, f))
                scale = np.sqrt(c ** 2 + f ** 2)

                referencePointRotation[j] = np.rad2deg(
                    np.arctan2(c_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0],
                               f_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]))
                referencePointScale[j] = np.sqrt(
                    c_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0] ** 2 +
                    f_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0] ** 2)

            if len(rot.shape) < 2:
                rot = np.ones(x_mesh.shape) * rot
                scale = np.ones(x_mesh.shape) * scale

            print('%s-rotation at projection reference point: %1.8f = %1.4f arcsec' % (
            axis, referencePointRotation[j], referencePointRotation[j] * u.deg.to(u.arcsecond)))
            print('%s-scale    at projection reference point: %1.8f' % (axis, referencePointScale[j]))

            origin = (referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])
            rotated_x, rotated_y = rotate(origin, x_mesh, y_mesh, np.deg2rad(rot))
            #             Offset due to rotation
            U_rot, V_rot = rotated_x - x_mesh, rotated_y - y_mesh

            # total offset due to k=4
            mode = 4
            xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh, y_mesh,
                                                          partial_mode=mode, includeAllHigherOrders=False)
            U_tot, V_tot = xp - x_mesh, yp - y_mesh

            # difference = scale offset
            U_scale, V_scale = U_tot - U_rot, V_tot - V_rot

            UU = np.zeros((3, U_scale.shape[0], U_scale.shape[1]))
            UU[0, :, :] = U_tot
            UU[1, :, :] = U_rot
            UU[2, :, :] = U_scale
            VV = np.zeros((3, V_scale.shape[0], V_scale.shape[1]))
            VV[0, :, :] = V_tot
            VV[1, :, :] = V_rot
            VV[2, :, :] = V_scale

            titles = np.array(['Total offset', 'Rotation offset', 'Scale offset'])
            for iii in [0, 1, 2]:
                ax = pl.subplot(3, 2, iii * 2 + j + 1)

                U = np.ma.masked_array(UU[iii, :, :], mask=~goodMeshMask)
                V = np.ma.masked_array(VV[iii, :, :], mask=~goodMeshMask)

                if axis == 'X':
                    z = U
                elif axis == 'Y':
                    z = V
                scale_min = np.min(z)
                scale_max = np.max(z)

                pl.pcolor(x_mesh * xy_scale, y_mesh * xy_scale, z, cmap='RdBu', vmin=scale_min, vmax=scale_max)

                #                 sc = pl.scatter(x_mesh.flatten()[goodMask], y_mesh.flatten()[goodMask], c=z[goodMask], cmap=cm,edgecolors='face',alpha=1.0,s=4) # , vmin=0, vmax=20, s=35
                #                 pl.colorbar(sc)
                pl.text(0.05, 0.07, 'Median %3.2f\nRMS %3.2f' % (np.ma.median(z), np.ma.std(z)), ha='left', va='center',
                        transform=ax.transAxes)
                pl.title('%s of k=4 terms in %s' % (titles[iii], axis))
                pl.axis('equal')
                pl.xlabel(xlabl)
                pl.ylabel(ylabl)
                cbar = pl.colorbar()
                cbar.ax.get_yaxis().labelpad = 10
                cbar.set_label('Amplitude (mas)', rotation=270)

            # distanceFromRefPoint = np.sqrt( (x_mesh - referencePointForProjection_Pix[0])**2 + (y_mesh - referencePointForProjection_Pix[1])**2 )
            #           offsetDueToRotation = distanceFromRefPoint *  * scale
            #
            #             rot = rotfunc(x_mesh,y_mesh)[0][0]
            #             scale = rot
            #             scale = wfunc(x_mesh,y_mesh)[0][0]
            #             scale_mean = np.mean(scale)
            #
            #             scale -= referencePointRotation[j]
            #             scale_min = np.min(scale)
            #             scale_max = np.max(scale)
            #             1/0
            #             pl.pcolor(x_mesh*xy_scale, y_mesh*xy_scale, scale, cmap='RdBu', vmin=scale_min, vmax=scale_max)
            #             pl.plot(referencePointForProjection_Pix[0],referencePointForProjection_Pix[1],'ko')
            #             if crossterm==0:
            #                 crossStr = ''
            #             else:
            #                 crossStr = 'crossterm'
            #             pl.title('%s-rotation reference=%1.6f deg'%(axis,referencePointRotation[j]))
            #             pl.axis('equal')
            #             pl.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
            #             pl.xlabel(xlabl)         pl.ylabel(ylabl)
            #             pl.text(0.05,0.05,'Median %1.1e\nRMS %1.1e' % (np.ma.median(scale),np.ma.std(scale)), ha='left', va='center', transform=ax.transAxes)
            #             pl.colorbar(format='%1.1e')
            pl.show()
        fig.tight_layout(h_pad=0.0)
        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_rotation.pdf' % (nameSeed))
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    def plotLinearTerms(self, evaluation_frame_number, outDir, nameSeed, referencePointForProjection_Pix, save_plot=1,
                        xy_unit='undefined', xy_scale=1.):
        # plot k=4 terms and split them up into contributions from global rotation, global scale, ans on- and off-axis skew
        ii = evaluation_frame_number
        ix = np.where(self.colNames == 'x')[0][0]
        iy = np.where(self.colNames == 'y')[0][0]
        xlabl = 'X (%s)' % xy_unit
        ylabl = 'Y (%s)' % xy_unit

        P = sympy.symbols('p0:%d' % self.Nalm)
        Cs, polynomialTermOrder = bivariate_polynomial(x, y, self.k, verbose=0)
        partialMode = 4
        includeAllHigherOrders = False
        polynomial = self.get_polynomial(Cs, P, partial_mode=partialMode, include_all_higher_orders=includeAllHigherOrders)

        points = np.array(self.C[[1, 2],], dtype='float32').T
        path = self.getHullPath(points)

        axes = ['X', 'Y']
        x_min = np.min(self.C[1,])  # ximinusx0
        x_max = np.max(self.C[1,])
        y_min = np.min(self.C[2,])
        y_max = np.max(self.C[2,])

        Ngrid = 100
        xx = np.linspace(x_min, x_max, Ngrid)
        yy = np.linspace(y_min, y_max, Ngrid)
        x_mesh, y_mesh = np.meshgrid(xx, yy)

        goodMeshMask = path.contains_points(np.array([x_mesh.flatten(), y_mesh.flatten()]).T).reshape(x_mesh.shape)
        x_mesh = np.ma.masked_array(x_mesh, mask=~goodMeshMask)
        y_mesh = np.ma.masked_array(y_mesh, mask=~goodMeshMask)

        b_func, c_func, e_func, f_func = self.getPartialDerivatives(polynomial, ii)

        b = b_func(x_mesh, y_mesh)[0][0]
        c = c_func(x_mesh, y_mesh)[0][0]
        e = e_func(x_mesh, y_mesh)[0][0]
        f = f_func(x_mesh, y_mesh)[0][0]

        xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis, skew_offaxis, rotation2 = computeRotScaleSkewFromPolyCoeff(
            b, c, e, f)

        b_ref = b_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]
        c_ref = c_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]
        e_ref = e_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]
        f_ref = f_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]

        xmag_ref, ymag_ref, xrotation_ref, yrotation_ref, rotation_ref, skew_ref, relative_scale_ref, skew_onaxis_ref, skew_offaxis_ref, rotation2_ref = computeRotScaleSkewFromPolyCoeff(
            b_ref, c_ref, e_ref, f_ref)

        referencePointScale = relative_scale_ref
        referencePointRotation = rotation2_ref
        print('Global scale    at projection reference point: %1.8f' % (referencePointScale))
        print('Global rotation at projection reference point: %1.8f' % (referencePointRotation))

        # compute displacements assuming on- and off-axis skew are zero
        # first rotation only
        #         b_simu,c_simu,e_simu,f_simu = computePolyCoeffFromRotScaleSkew(relative_scale_ref,relative_scale_ref,np.deg2rad(rotation2_ref),np.deg2rad(rotation2_ref))
        b_simu, c_simu, e_simu, f_simu = computePolyCoeffFromRotScaleSkew(1., 1., np.deg2rad(rotation2_ref),
                                                                          np.deg2rad(rotation2_ref))
        P_simu = sympy.symbols('p0:%d' % 3)
        Cs_simu, polynomialTermOrder_simu = bivariate_polynomial(x, y, 4, verbose=0)
        polynomial_simu = self.get_polynomial(Cs_simu, P_simu, partial_mode=0)
        replacements_x = '("p0",0.),("p1",b_simu),("p2",c_simu)'
        replacements_y = '("p0",0.),("p1",e_simu),("p2",f_simu)'
        func_x = polynomial_simu.subs(eval('[' + replacements_x + ']'))
        func_y = polynomial_simu.subs(eval('[' + replacements_y + ']'))
        wfunc_x = lambdify((x, y), func_x, 'numpy')
        wfunc_y = lambdify((x, y), func_y, 'numpy')
        U_rot, V_rot = wfunc_x(x_mesh, y_mesh)[0][0] - x_mesh, wfunc_y(x_mesh, y_mesh)[0][0] - y_mesh

        # now scale only
        b_simu, c_simu, e_simu, f_simu = computePolyCoeffFromRotScaleSkew(relative_scale_ref, relative_scale_ref, 0.,
                                                                          0.)
        func_x = polynomial_simu.subs(eval('[' + replacements_x + ']'))
        func_y = polynomial_simu.subs(eval('[' + replacements_y + ']'))
        wfunc_x = lambdify((x, y), func_x, 'numpy')
        wfunc_y = lambdify((x, y), func_y, 'numpy')
        U_scale, V_scale = wfunc_x(x_mesh, y_mesh)[0][0] - x_mesh, wfunc_y(x_mesh, y_mesh)[0][0] - y_mesh

        # total offset due to k=4
        mode = 4
        xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh, y_mesh,
                                                      partial_mode=mode, includeAllHigherOrders=False)

        U_tot, V_tot = xp - x_mesh, yp - y_mesh

        U_nonlin = U_tot - U_rot - U_scale
        V_nonlin = V_tot - V_rot - V_scale

        UU = np.zeros((4, U_scale.shape[0], U_scale.shape[1]))
        UU[0, :, :] = U_tot
        UU[1, :, :] = U_rot
        UU[2, :, :] = U_scale
        UU[3, :, :] = U_nonlin
        VV = np.zeros((4, V_scale.shape[0], V_scale.shape[1]))
        VV[0, :, :] = V_tot
        VV[1, :, :] = V_rot
        VV[2, :, :] = V_scale
        VV[3, :, :] = V_nonlin

        fig = pl.figure(figsize=(10, 16), facecolor='w', edgecolor='k')
        pl.clf()
        titles = np.array(['Total offset', 'Rotation offset', 'Scale offset', 'Skew offset'])
        for j, axis in enumerate(axes):
            for iii in [0, 1, 2, 3]:
                ax = pl.subplot(4, 2, iii * 2 + j + 1)

                U = np.ma.masked_array(UU[iii, :, :], mask=~goodMeshMask)
                V = np.ma.masked_array(VV[iii, :, :], mask=~goodMeshMask)

                if axis == 'X':
                    z = U
                elif axis == 'Y':
                    z = V
                scale_min = np.min(z)
                scale_max = np.max(z)

                pl.pcolor(x_mesh * xy_scale, y_mesh * xy_scale, z, cmap='RdBu', vmin=scale_min, vmax=scale_max)
                pl.text(0.05, 0.07, 'Median %3.2f\nRMS %3.2f' % (np.ma.median(z), np.ma.std(z)), ha='left', va='center',
                        transform=ax.transAxes)
                pl.title('%s of k=4 terms in %s' % (titles[iii], axis))
                pl.axis('equal')
                pl.xlabel(xlabl)
                pl.ylabel(ylabl)
                cbar = pl.colorbar()
                cbar.ax.get_yaxis().labelpad = 10
                cbar.set_label('Amplitude (mas)', rotation=270)

        fig.tight_layout(h_pad=0.0)
        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_LinearTerms.pdf' % (nameSeed))
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

        # now figure showing everythin that is not offset, global scale, or global rotation
        # total offset due to k>4
        mode = 4
        xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh, y_mesh,
                                                      partial_mode=mode, includeAllHigherOrders=True)
        U_tot, V_tot = xp - x_mesh, yp - y_mesh

        #         U_nonlin = U_tot - U_rot - U_scale
        #         V_nonlin = V_tot - V_rot - V_scale
        U_nonlin = U_tot - U_rot
        V_nonlin = V_tot - V_rot

        UU = np.zeros((1, U_scale.shape[0], U_scale.shape[1]))
        UU[0, :, :] = U_nonlin
        VV = np.zeros((1, V_scale.shape[0], V_scale.shape[1]))
        VV[0, :, :] = V_nonlin

        fig = pl.figure(figsize=(12, 5), facecolor='w', edgecolor='k')
        pl.clf()
        titles = np.array(['', 'Rotation offset', 'Scale offset', 'Skew offset'])
        for j, axis in enumerate(axes):
            for iii in [0]:
                ax = pl.subplot(1, 2, iii * 2 + j + 1)

                U = np.ma.masked_array(UU[iii, :, :], mask=~goodMeshMask)
                V = np.ma.masked_array(VV[iii, :, :], mask=~goodMeshMask)

                if axis == 'X':
                    z = U
                elif axis == 'Y':
                    z = V
                scale_min = np.min(z)
                scale_max = np.max(z)

                pl.pcolor(x_mesh * xy_scale, y_mesh * xy_scale, z, cmap='RdBu', vmin=scale_min, vmax=scale_max)
                pl.text(0.05, 0.07, 'Median %3.2f\nRMS %3.2f' % (np.ma.median(z), np.ma.std(z)), ha='left', va='center',
                        transform=ax.transAxes)
                #                 pl.title('%s of k=4 terms in %s'%(titles[iii],axis))
                pl.axis('equal')
                pl.xlabel(xlabl)
                pl.ylabel(ylabl)
                cbar = pl.colorbar()
                cbar.ax.get_yaxis().labelpad = 10
                cbar.set_label('Amplitude (mas)', rotation=270)

        fig.tight_layout(h_pad=0.0)
        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_nonLinearTerms.pdf' % (nameSeed))
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    def getPartialDerivatives(self, polynomial, evaluation_frame_number):
        ii = evaluation_frame_number
        replacements_X = (','.join(['("p%d",self.Alm[%d,%d])' % (d, ii, d) for d in range(self.Nalm)]))
        replacements_Y = (','.join(['("p%d",self.Alm[%d,self.Nalm+%d])' % (d, ii, d) for d in range(self.Nalm)]))
        dfexpr_b = sympy.diff(polynomial, x)
        dfexpr_c = sympy.diff(polynomial, y)
        dfexpr_e = sympy.diff(polynomial, x)
        dfexpr_f = sympy.diff(polynomial, y)

        dffunc_b = dfexpr_b.subs(eval('[' + replacements_X + ']'))
        dffunc_c = dfexpr_c.subs(eval('[' + replacements_X + ']'))
        dffunc_e = dfexpr_e.subs(eval('[' + replacements_Y + ']'))
        dffunc_f = dfexpr_f.subs(eval('[' + replacements_Y + ']'))

        b_func = lambdify((x, y), dffunc_b, 'numpy')
        c_func = lambdify((x, y), dffunc_c, 'numpy')
        e_func = lambdify((x, y), dffunc_e, 'numpy')
        f_func = lambdify((x, y), dffunc_f, 'numpy')

        return b_func, c_func, e_func, f_func

    #     def evalFunctionOnMesh(self,func,xx,yy):
    #
    #     if len(wfunc_x(xx,yy).shape) == 4:
    #             return_values = wfunc_x(xx,yy)[0][0],wfunc_y(xx,yy)[0][0]
    #         elif len(wfunc_x(xx,yy).shape) == 2:
    #             ret_x = np.ones(xx.shape) * wfunc_x(xx,yy)[0][0]
    #             ret_y = np.ones(yy.shape) * wfunc_y(xx,yy)[0][0]
    #             return_values = ret_x,ret_y
    #
    def plotScale(self, evaluation_frame_number, outDir, nameSeed, referencePointForProjection_Pix, save_plot=1,
                  xy_unit='undefined', xy_scale=1.):
        ii = evaluation_frame_number
        ix = np.where(self.colNames == 'x')[0][0]
        iy = np.where(self.colNames == 'y')[0][0]
        xlabl = 'X (%s)' % xy_unit
        ylabl = 'Y (%s)' % xy_unit

        P = sympy.symbols('p0:%d' % self.Nalm)
        Cs, polynomialTermOrder = bivariate_polynomial(x, y, self.k, verbose=0)
        polynomial = Matrix(Cs.T) * Matrix(P)

        points = np.array(self.C[[1, 2],], dtype='float32').T
        hull = scipy.spatial.ConvexHull(points)
        path = matplotlib.path.Path(points[hull.vertices,])

        axes = ['x', 'y']
        x_min = np.min(self.C[1,])  # ximinusx0
        x_max = np.max(self.C[1,])
        y_min = np.min(self.C[2,])
        y_max = np.max(self.C[2,])

        Ngrid = 100
        xx = np.linspace(x_min, x_max, Ngrid)
        yy = np.linspace(y_min, y_max, Ngrid)
        #             xx = self.p[ii,:,ix]
        #             yy = self.p[ii,:,iy]
        #         goodIdx = np.where(path.contains_points(np.array([xx,yy]).T))[0]
        #         x_mesh, y_mesh = np.meshgrid(xx[goodIdx], yy[goodIdx])
        x_mesh, y_mesh = np.meshgrid(xx, yy)

        goodMeshMask = path.contains_points(np.array([x_mesh.flatten(), y_mesh.flatten()]).T).reshape(x_mesh.shape)
        x_mesh = np.ma.masked_array(x_mesh, mask=~goodMeshMask)
        y_mesh = np.ma.masked_array(y_mesh, mask=~goodMeshMask)

        if 1 == 1:  # plot global scale according to HST/Jay definition
            partialMode = 0  # full polynomial
            polynomial = self.get_polynomial(Cs, P, partial_mode=partialMode)
            b_func, c_func, e_func, f_func = self.getPartialDerivatives(polynomial, ii)

            b = b_func(x_mesh, y_mesh)[0][0]
            c = c_func(x_mesh, y_mesh)[0][0]
            e = e_func(x_mesh, y_mesh)[0][0]
            f = f_func(x_mesh, y_mesh)[0][0]

            xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis, skew_offaxis, rotation2 = computeRotScaleSkewFromPolyCoeff(
                b, c, e, f)

            b_ref = b_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]
            c_ref = c_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]
            e_ref = e_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]
            f_ref = f_func(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])[0][0]

            xmag_ref, ymag_ref, xrotation_ref, yrotation_ref, rotation_ref, skew_ref, relative_scale_ref, skew_onaxis_ref, skew_offaxis_ref, rotation2_ref = computeRotScaleSkewFromPolyCoeff(
                b_ref, c_ref, e_ref, f_ref)

            referencePointScale = relative_scale_ref
            print('Global scale at projection reference point: %1.8f' % (referencePointScale))

            scale = relative_scale
            scale -= referencePointScale
            scale_min = np.min(scale)
            scale_max = np.max(scale)

            fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
            pl.clf()
            pl.pcolor(x_mesh * xy_scale, y_mesh * xy_scale, scale, cmap='RdBu', vmin=scale_min, vmax=scale_max)
            pl.plot(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1], 'ko')

            pl.title('Global scale, reference=%1.8f' % (referencePointScale))
            pl.axis('equal')
            pl.xlabel(xlabl)
            pl.ylabel(ylabl)
            ax = pl.gca()
            pl.text(0.05, 0.05, 'Median %1.1e\nRMS %1.1e' % (np.ma.median(scale), np.ma.std(scale)), ha='left',
                    va='center', transform=ax.transAxes)
            pl.colorbar(format='%1.1e')
            pl.show()
            fig.tight_layout(h_pad=0.0)
            if save_plot == 1:
                figName = os.path.join(outDir, '%s_globalscale.pdf' % (nameSeed))
                pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

        for crossterm in [0, 1]:
            fig = pl.figure(figsize=(12, 5), facecolor='w', edgecolor='k')
            pl.clf()
            referencePointScale = np.zeros(2)
            for j, axis in enumerate(axes):
                ax = pl.subplot(1, 2, j + 1)
                if axis == 'x':
                    if crossterm == 0:
                        dfexpr = sympy.diff(polynomial, x)
                    else:
                        dfexpr = sympy.diff(polynomial, y)
                    d0 = 0
                elif axis == 'y':
                    if crossterm == 0:
                        dfexpr = sympy.diff(polynomial, y)
                    else:
                        dfexpr = sympy.diff(polynomial, x)
                    d0 = self.Nalm

                replacements = (','.join(['("p%d",self.Alm[%d,%d])' % (d, ii, d0 + d) for d in range(self.Nalm)]))
                dffunc = dfexpr.subs(eval('[' + replacements + ']'))
                wfunc = lambdify((x, y), dffunc, 'numpy')

                referencePointScale[j] = wfunc(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1])
                print('%s-scale at projection reference point: %1.8f' % (axis, referencePointScale[j]))

                scale = wfunc(x_mesh, y_mesh)[0][0]
                #             scale_mean = np.mean(scale)
                scale -= referencePointScale[j]
                scale_min = np.min(scale)
                scale_max = np.max(scale)

                pl.pcolor(x_mesh * xy_scale, y_mesh * xy_scale, scale, cmap='RdBu', vmin=scale_min, vmax=scale_max)
                pl.plot(referencePointForProjection_Pix[0], referencePointForProjection_Pix[1], 'ko')
                if crossterm == 0:
                    crossStr = ''
                else:
                    crossStr = 'crossterm'
                pl.title('%s-scale %s reference=%1.8f' % (axis, crossStr, referencePointScale[j]))
                pl.axis('equal')
                #             pl.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
                pl.xlabel(xlabl)
                pl.ylabel(ylabl)
                pl.text(0.05, 0.05, 'Median %1.1e\nRMS %1.1e' % (np.ma.median(scale), np.ma.std(scale)), ha='left',
                        va='center', transform=ax.transAxes)
                pl.colorbar(format='%1.1e')
                pl.show()
            fig.tight_layout(h_pad=0.0)
            pl.show()
            if save_plot == 1:
                figName = os.path.join(outDir, '%s_scale_%s.pdf' % (nameSeed, crossStr))
                pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    def get_polynomial(self, Cs, P, partial_mode=0, include_all_higher_orders=True):
        """Return a polynomial model.

        Parameters
        ----------
        Cs
        P
        partial_mode : int
            If zero, the full polynomial includign all terms is computed
        include_all_higher_orders : bool

        Returns
        -------
        polynomial :

        """
        if partial_mode == 0:
            # full model
            polynomial = Matrix(Cs.T) * Matrix(P)
        else:
            minMode = partial_mode - 2
            maxMode = partial_mode
            startIndex = np.int(minMode * (minMode + 2) / 8)
            stopIndex = np.int(maxMode * (maxMode + 2) / 8)
            print('minMode = %d, Startindex = %d, maxMode = %d, stopindex = %d' % (
            minMode, startIndex, maxMode, stopIndex))
            if include_all_higher_orders:
                polynomial = Matrix(Cs[startIndex:].T) * Matrix(P[startIndex:])
            else:
                polynomial = Matrix(Cs[startIndex:stopIndex].T) * Matrix(P[startIndex:stopIndex])

        return polynomial

    def apply_polynomial_transformation(self, evaluation_frame_number, xx_in, yy_in, partial_mode=0, includeAllHigherOrders=False):
        """Apply the polynomial model defined by the coefficients stored in self.Alm to the input coordinates xx,yy.

        Attention when using reduced coordinates!

        Parameters
        ----------
        evaluation_frame_number
        xx
        yy
        partial_mode : int
            By default, this is 0 and the full polynomial is applied.
        includeAllHigherOrders

        Returns
        -------

        """
        ii = evaluation_frame_number

        xx = copy.deepcopy(xx_in)
        yy = copy.deepcopy(yy_in)

        correct_for_reference_point = True
        if correct_for_reference_point:
            # subtract reference point
            referencePositionX = self.referencePoint[np.int(ii), 0]
            referencePositionY = self.referencePoint[np.int(ii), 1]

            xx -= referencePositionX
            yy -= referencePositionY

        P = sympy.symbols('p0:%d' % self.Nalm)
        Cs, polynomialTermOrder = bivariate_polynomial(sympy_x, sympy_y, self.k, verbose=0)

        replacements_x = (','.join(['("p%d",self.Alm[%d,%d])' % (d, ii, d) for d in range(self.Nalm)]))
        replacements_y = (','.join(['("p%d",self.Alm[%d,self.Nalm+%d])' % (d, ii, d) for d in range(self.Nalm)]))

        polynomial = self.get_polynomial(Cs, P, partial_mode=partial_mode, include_all_higher_orders=includeAllHigherOrders)

        func_x = polynomial.subs(eval('[' + replacements_x + ']'))
        func_y = polynomial.subs(eval('[' + replacements_y + ']'))
        wfunc_x = lambdify((x, y), func_x, 'numpy')
        wfunc_y = lambdify((x, y), func_y, 'numpy')

        if len(wfunc_x(xx, yy).shape) == 4:
            return_values = wfunc_x(xx, yy)[0][0], wfunc_y(xx, yy)[0][0]
        elif len(wfunc_x(xx, yy).shape) == 3:
            return_values = wfunc_x(xx, yy)[0][0], wfunc_y(xx, yy)[0][0]
        elif len(wfunc_x(xx, yy).shape) == 2:
            ret_x = np.ones(xx.shape) * wfunc_x(xx, yy)[0][0]
            ret_y = np.ones(yy.shape) * wfunc_y(xx, yy)[0][0]
            return_values = ret_x, ret_y

        if correct_for_reference_point:
            return_values = return_values[0] + referencePositionX, return_values[1] + referencePositionY


        return return_values


    def plotDistortion(self, evaluation_frame_number, outDir, nameSeed, save_plot=1,
                       xy_unit='undefined', xy_scale=1., detailed_plot_k=8):
        ii = evaluation_frame_number
        ix = np.where(self.colNames == 'x')[0][0]
        iy = np.where(self.colNames == 'y')[0][0]

        xlabl = 'X (%s)' % xy_unit
        ylabl = 'Y (%s)' % xy_unit

        P = sympy.symbols('p0:%d' % self.Nalm)
        Cs, polynomialTermOrder = bivariate_polynomial(x, y, self.k, verbose=0)

        points = np.array(self.C[[1, 2],], dtype='float32').T
        hull = scipy.spatial.ConvexHull(points)
        path = matplotlib.path.Path(points[hull.vertices,])

        x_min = np.min(self.C[1,])  # ximinusx0
        x_max = np.max(self.C[1,])
        y_min = np.min(self.C[2,])
        y_max = np.max(self.C[2,])

        Ngrid = 50
        xx = np.linspace(x_min, x_max, Ngrid)
        yy = np.linspace(y_min, y_max, Ngrid)
        x_mesh, y_mesh = np.meshgrid(xx, yy)

        replacements_x = (','.join(['("p%d",self.Alm[%d,%d])' % (d, ii, d) for d in range(self.Nalm)]))
        replacements_y = (','.join(['("p%d",self.Alm[%d,self.Nalm+%d])' % (d, ii, d) for d in range(self.Nalm)]))

        goodMeshMask = path.contains_points(np.array([x_mesh.flatten(), y_mesh.flatten()]).T).reshape(x_mesh.shape)

        if self.k == detailed_plot_k:
            #             cm = pl.cm.get_cmap('RdYlBu')
            cm = pl.cm.get_cmap('RdBu')
            axes = ['X', 'Y']
            modes = np.array([44, 4, 6])
            for jj, mode in enumerate(modes):
                if mode == 4:
                    xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh,
                                                                  y_mesh, partial_mode=mode,
                                                                  includeAllHigherOrders=False)
                    xp_0, yp_0 = x_mesh, y_mesh
                elif mode == 44:
                    modetmp = 4
                    xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh,
                                                                  y_mesh, partial_mode=modetmp,
                                                                  includeAllHigherOrders=True)
                    xp_0, yp_0 = x_mesh, y_mesh
                elif mode == 6:
                    xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh,
                                                                  y_mesh, partial_mode=mode,
                                                                  includeAllHigherOrders=True)
                    xp_0, yp_0 = 0., 0.
                U = np.ma.masked_array(xp, mask=~goodMeshMask) - xp_0  # to show offsets
                V = np.ma.masked_array(yp, mask=~goodMeshMask) - yp_0

                if mode == 44:
                    print(
                        'plotDistortion: terms of {0:d} and higher  max/min/rms correction in X is {1:1.2f}/{2:1.2f}/{5:1.2f}  max/min correction in Y is {3:1.2f}/{4:1.2f}/{6:1.2f}'.format(
                            mode, np.ma.max(U), np.ma.min(U), np.ma.max(V), np.ma.min(V), np.ma.std(U), np.ma.std(V)))

                fig = pl.figure(figsize=(12, 5), facecolor='w', edgecolor='k')
                pl.clf()
                for j, axis in enumerate(axes):
                    ax = pl.subplot(1, 2, j + 1)
                    if axis == 'X':
                        z = U
                    elif axis == 'Y':
                        z = V
                    scale_min = np.min(z)
                    scale_max = np.max(z)

                    #                     my_cmap = 'bone'
                    my_cmap = 'RdBu'
                    pl.pcolor(x_mesh * xy_scale, y_mesh * xy_scale, z, cmap=my_cmap, vmin=scale_min, vmax=scale_max)

                    #                 sc = pl.scatter(x_mesh.flatten()[goodMask], y_mesh.flatten()[goodMask], c=z[goodMask], cmap=cm,edgecolors='face',alpha=1.0,s=4) # , vmin=0, vmax=20, s=35
                    #                 pl.colorbar(sc)
                    pl.text(0.05, 0.05, 'Median %3.2f\nRMS %3.2f' % (np.ma.median(z), np.ma.std(z)), ha='left',
                            va='center', transform=ax.transAxes)
                    if mode == 4:
                        pl.title('Distortion in %s of k=4 terms' % (axis))
                    elif mode == 44:
                        pl.title('Distortion in %s of k>2 terms' % (axis))
                    else:
                        pl.title('Distortion in %s of k>4 terms' % (axis))
                    pl.axis('equal')
                    pl.xlabel(xlabl)
                    pl.ylabel(ylabl)
                    cbar = pl.colorbar()
                    cbar.ax.get_yaxis().labelpad = 10
                    cbar.set_label('Amplitude (mas)', rotation=270)
                    #             pl.colorbar(format='%1.1e')
                fig.tight_layout(h_pad=0.0)
                pl.show()
                if save_plot == 1:
                    figName = os.path.join(outDir, '%s_mode%d_distortionSkyNew.pdf' % (nameSeed, mode))
                    pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)
                    #             1/0

        modes = np.arange(self.k + 1)[::2]
        fig = pl.figure(figsize=(12, np.ceil(len(modes) / 3.) * 3), facecolor='w', edgecolor='k')
        pl.clf()
        for jj, mode in enumerate(modes):
            pl.subplot(np.ceil(len(modes) / 3.), 3, jj + 1)

            if mode == 0:  # show total distortion
                xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh, y_mesh, partial_mode=mode)
                xp_0, yp_0 = x_mesh, y_mesh
                lbl = 'total'
            else:
                xp, yp = self.apply_polynomial_transformation(evaluation_frame_number, x_mesh, y_mesh, partial_mode=mode)
                if mode == 4:
                    xp_0, yp_0 = x_mesh, y_mesh
                else:
                    xp_0, yp_0 = 0., 0.
                lbl = 'k=%d terms' % (mode)

            U = np.ma.masked_array(xp, mask=~goodMeshMask) - xp_0  # to show offsets
            V = np.ma.masked_array(yp, mask=~goodMeshMask) - yp_0

            #             print('plotDistortion: {0:s}  max/min correction in X is {1:1.2f}/{2:1.2f}  max/min correction in Y is {3:1.2f}/{4:1.2f}'.format(lbl,np.ma.max(U),np.ma.min(U),np.ma.max(V),np.ma.min(V)))
            print(
                'plotDistortion: {0:d}  max/min/rms correction in X is {1:1.2f}/{2:1.2f}/{5:1.2f}  max/min correction in Y is {3:1.2f}/{4:1.2f}/{6:1.2f}'.format(
                    mode, np.ma.max(U), np.ma.min(U), np.ma.max(V), np.ma.min(V), np.ma.std(U), np.ma.std(V)))

            #             pl.quiver(x_mesh*xy_scale,y_mesh*xy_scale, U, V, angles='xy')
            pl.streamplot(xx * xy_scale, yy * xy_scale, U, V)

            maxlim = np.max([x_mesh, y_mesh]) * xy_scale
            pl.xlim((-maxlim, maxlim))
            pl.ylim((-maxlim, maxlim))
            pl.axis('square')
            pl.xlabel(xlabl)
            pl.ylabel(ylabl)
            pl.title('Distortion model (%s)' % lbl)
        fig.tight_layout(h_pad=0.0)
        pl.show()
        if save_plot == 1:
            figName = os.path.join(outDir, '%s_distortion.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    def correctCatalogForDistortion(self, evaluation_frame_number, ra, dec, referencePointForProjection_RADec, scale):
        ii = evaluation_frame_number
        x_in, y_in = RADec2Pix_TAN(ra, dec, referencePointForProjection_RADec[0], referencePointForProjection_RADec[1],
                                   scale)
        # P = sympy.symbols('p0:%d' % self.Nalm)
        # Cs, polynomialTermOrder = bivariate_polynomial(x, y, self.k, verbose=0)
        x_corr, y_corr = self.apply_polynomial_transformation(evaluation_frame_number, x_in, y_in,
                                                              partial_mode=0)
        ra_corr, dec_corr = Pix2RADec_TAN(x_corr, y_corr, referencePointForProjection_RADec[0],
                                          referencePointForProjection_RADec[1], scale)
        return ra_corr, dec_corr


    def identify_poorly_behaved_stars(self, exclude_target=True, absolute_threshold=10, show_plot=False):
        """Return list of stars that violate quality criteria in terms of residual amplitude after the distortion fit.

        Returns
        -------

        """
        number_of_frames = self.Alm.shape[0]
        # number_of_stars = len(self.resx[0].residuals)
        number_of_stars = self.p.shape[1]
        coord = np.arange(number_of_frames)

        bad_star_index = []

        if show_plot:
            pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
            pl.clf()
        for j in np.arange(number_of_stars):
            if (j == 0) & (exclude_target):  # skip target
                continue
            index = np.where(self.p[:, j, self.colNames.tolist().index('artificial')] == 0)[0]
            residuals_x = np.array([self.resx[i].residuals[j] for i in range(number_of_frames)])
            residuals_y = np.array([self.resy[i].residuals[j] for i in range(number_of_frames)])
            # if (np.abs(np.mean(residuals_x[index])) > threshold) or (np.abs(np.mean(residuals_y[index])) > threshold):
            if (np.any(np.abs(residuals_x[index]) > absolute_threshold)) or (np.any(np.abs(residuals_y[index]) > absolute_threshold)):
                bad_star_index.append(j)
                continue
            if show_plot:
                pl.subplot(2, 1, 1)
                pl.plot(coord[index], residuals_x[index])
                pl.subplot(2, 1, 2)
                pl.plot(coord[index], residuals_y[index])
        if show_plot:
            pl.xlabel('Frame number')
            pl.subplot(2, 1, 1)
            pl.title('Distortion-fit residuals of {} well-behaved stars (|O-C|<{:2.2f})'.format(number_of_stars-len(bad_star_index), absolute_threshold))
            pl.show()

        # plot RMS
        if 0:
            pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
            pl.clf()

            for i in np.arange(number_of_frames):
                index = np.where(self.p[i, :, self.colNames.tolist().index('artificial')] == 0)[0]
                # 1/0
                index = index[index!=0]
                # index = np.append(index, 0) # exclude target
                # residuals_x = np.array(self.resx[i].residuals)
                # residuals_y = np.array(self.resy[i].residuals)
                residuals_y = np.array([self.resy[i].residuals[j] for j in index])
                residuals_x = np.array([self.resx[i].residuals[j] for j in index])
                # residuals_y = np.array([self.resy[i].residuals[j] for j in range(number_of_stars)])
                pl.subplot(2,1,1)
                # pl.plot(i, np.std(residuals_x[index]), 'bo')
                pl.plot(i, np.std(residuals_x), 'bo')
                pl.subplot(2,1,2)
                pl.plot(i, np.std(residuals_y), 'ro')
                # pl.plot(i, np.std(residuals_y[index]), 'ro')
            pl.xlabel('Frame number')
            pl.show()


        return bad_star_index


def bivariate_polynomial(xval, yval, k=None, degree=None, maxDegree=None, verbose=False):
    """
    Generate general polynomial evaluated at xval,yval for use in astrometric distortion calculations
    written 2016-12-05 J. Sahlmann, AURA/STScI

    cases that remain to be implemented:  k=None, degree = [degx,degy] and maxDegree is no None

    Parameters
    ----------
    xval : string
        user name
    myPsswd : string
        password
    commandString : string
        string to be sent to GACS

    """

    #   test for k is positive even integer, #   degree is 2 list # maxDegree is positive integer

    if k is not None:
        polyDegree = k / 2 - 1
        maxDegree = polyDegree
        degree = [polyDegree, polyDegree]
        Nfree = k * (k + 2) / 8
        if verbose:
            print('Standard astrometry polynomial: mode k=%d, order=%d, maxDegree=%d, number of free parameters=%d' % (
                k, polyDegree, maxDegree, Nfree))

        gp = np.polynomial.polynomial.polyvander2d(xval, yval, degree)

        # eliminate terms that have total degree > maxDegree
        from sympy.abc import x, y
        xa = np.polynomial.polynomial.polyvander2d(x, x, degree)[0]
        #         totalDegree = np.array([t.args[0] for t in xa])
        totalDegree = np.array([t.as_base_exp()[1] for t in xa])

        goodIndex = np.array([0])

        # return array of indices that correspond to terms with totalDegree <= maxDegree and that are sorted according to JWST SIAF standard, i.e. 1,x,y,x**2,xy,y**2,x**3,x**2y, ...
        if maxDegree > 0:
            for totDeg in np.arange(1, maxDegree + 1):
                sortedIndex = np.where(totalDegree == totDeg)[0][::-1]
                if totDeg == 1:
                    # remove last index in this array because it is the unit term
                    sortedIndex = sortedIndex[0:-1]
                goodIndex = np.hstack((goodIndex, sortedIndex))
                # this returns an unsorted array
                #             goodIndex = np.where( totalDegree <= maxDegree )[0]

        xb = np.polynomial.polynomial.polyvander2d(x, y, degree)[0]
        if verbose:
            print('Order of the returned array/matrix', xb[goodIndex])

        return np.mat(gp[:, goodIndex]).T, xb[goodIndex].astype(np.str)


def getRefStarAstrometricData(mp, targetId):
    # remove target from data array
    s = mp.p[:, np.where((mp.p[0, :, 6] != targetId))[0], :]
    return s


def getCleanedAstrometricData(mp, s, targetId):
    p = np.concatenate((mp.p[:, np.where((mp.p[0, :, 6] == targetId))[0], :], s), axis=1)
    mp.p = p
    targetIndex = 0
    return mp, targetIndex



def plot_distortion_statistics(lazAC, epoch_boundaries=None, show_plot=True, save_plot=False,
                               reference_frame_index=None, name_seed='', plot_dir='', coord=None,
                               parameters_to_plot=None, parameter_labels=None):
    """Make figures that show the evolution of the distortion fits stored in lazAC.

    Parameters
    ----------
    lazAC
    epoch_boundaries
    show_plot
    save_plot
    reference_frame_index
    name_seed
    plot_dir

    Returns
    -------

    """
    parameter_labels_default = {'Shift in X'     : 'Shift in X (arcsec)',
                        'Shift in Y'     : 'Shift in Y (arcsec)',
                        'Global Rotation': 'Global Rotation (deg)',
                        'Global Scale'   : 'Global Scale (unitless)',
                        'Rotation in X'  : 'Rotation in X (deg)',
                        'Rotation'  : 'Rotation (deg)',
                        'Rotation in Y'  : 'Rotation in Y (deg)',
                        'Scale in X'     : 'Scale in X (unitless)',
                        'Scale in Y'     : 'Scale in Y (unitless)',
                        'Skew'   : 'Skew (unitless)',
                        'On-axis Skew'   : 'On-axis Skew (unitless)',
                        'Off-axis Skew'  : 'Off-axis Skew (unitless)'}
    if parameter_labels is None:
        parameter_labels = parameter_labels_default


    if coord is None:
        coord = np.arange(lazAC.Alm.shape[0])
    values = False
    for exposure_number in range(lazAC.Alm.shape[0]):
        human_readable_solution_parameters = \
            compute_rot_scale_skew(lazAC, i=exposure_number)

        if values is False:
            values = human_readable_solution_parameters['values'][:, 0]
            uncert = human_readable_solution_parameters['values'][:, 1]
        else:
            values = np.vstack(
                (values, human_readable_solution_parameters['values'][:, 0]))
            uncert = np.vstack(
                (uncert, human_readable_solution_parameters['values'][:, 1]))

        val = Table(values, names=human_readable_solution_parameters['names'])
        unc = Table(uncert, names=['sigma_{}'.format(s) for s in
                                   human_readable_solution_parameters['names']])

    T = tablehstack((val, unc))

    if parameters_to_plot is not None:
        row_width = 7
        column_width = 2
    elif lazAC.useReducedCoordinates == 0:
        parameters_to_plot = ['Shift in X', 'Shift in Y', 'Global Rotation', 'Global Scale', 'On-axis Skew', 'Off-axis Skew']
        row_width = 7
        column_width = 2
    else:
        parameters_to_plot = human_readable_solution_parameters['names']
        row_width = 3
        column_width = 3
    n_panels = len(parameters_to_plot)
    n_figure_columns = 3
    n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))


    if show_plot:
        fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                figsize=(n_figure_rows * row_width, n_figure_columns * column_width),
                                facecolor='w', edgecolor='k', sharex=True, sharey=False,
                                squeeze=False)
        # axes_max = 0
        for jj, name in enumerate(parameters_to_plot):
            fig_row = jj % n_figure_rows
            fig_col = jj // n_figure_rows

            # fig_col = np.int(np.floor(jj/n_figure_columns))
            # fig_row = np.int(jj%n_figure_columns)

            axes[fig_row][fig_col].plot(coord, T[name], 'bo')
            axes[fig_row][fig_col].errorbar(coord, T[name],
                                            yerr=T['sigma_{}'.format(name)],
                                            fmt='none', ecolor='b')
            # axes[fig_row][fig_col].plot(coord[reference_index], T[name][
            # reference_index], 'go')
            if reference_frame_index is not None:
                axes[fig_row][fig_col].plot(coord[reference_frame_index],
                                            T[name][reference_frame_index], 'yo', color='0.5', ms=10)

            # axes[fig_row][fig_col].set_title(name)
            if fig_row == n_figure_rows - 1:
                axes[fig_row][fig_col].set_xlabel('Frame number')
            if parameter_labels is None:
                axes[fig_row][fig_col].set_ylabel(name)
            else:
                axes[fig_row][fig_col].set_ylabel(parameter_labels[name])

            if epoch_boundaries is not None:
                for boundary in epoch_boundaries:
                    axes[fig_row][fig_col].axvline(boundary, color='0.7', ls='--')

            if name == 'Global Scale':
                axes[fig_row][fig_col].axhline(1, ls='--', c='0.7')
            else:
                axes[fig_row][fig_col].axhline(0, ls='--', c='0.7')

        fig.tight_layout(h_pad=0.0)
        pl.show()

        if save_plot:
            figure_name = os.path.join(plot_dir, '{}_distortion_parameters_per_frame.pdf'.format(name_seed))
            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

    distortion_statistics = Table()
    distortion_statistics['rms_x'] = lazAC.rms[:, 0]
    distortion_statistics['rms_y'] = lazAC.rms[:, 1]
    distortion_statistics['n_stars'] = [lazAC.data[m]['number_of_stars_with_non_zero_weight'] for m in lazAC.data.keys()]

    n_panels = len(distortion_statistics.colnames)
    n_figure_columns = 1
    n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

    if show_plot:
        fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                figsize=(n_figure_columns * 10, n_figure_rows * 2),
                                facecolor='w', edgecolor='k', sharex=True, sharey=False,
                                squeeze=False)
        # axes_max = 0
        for jj, name in enumerate(distortion_statistics.colnames):
            fig_row = jj % n_figure_rows
            fig_col = jj // n_figure_rows

            # fig_col = np.int(np.floor(jj/n_figure_columns))
            # fig_row = np.int(jj%n_figure_columns)

            axes[fig_row][fig_col].plot(coord, distortion_statistics[name], 'bo')
            if reference_frame_index is not None:
                axes[fig_row][fig_col].plot(coord[reference_frame_index],
                                            distortion_statistics[name][reference_frame_index], 'yo', c='0.7', ms=10)

            if fig_row == n_figure_rows - 1:
                axes[fig_row][fig_col].set_xlabel('Frame number')
            axes[fig_row][fig_col].set_ylabel(name)

            if epoch_boundaries is not None:
                for boundary in epoch_boundaries:
                    axes[fig_row][fig_col].axvline(boundary, color='0.7', ls='--')

        fig.tight_layout(h_pad=0.0)
        pl.show()
        if save_plot:
            figure_name = os.path.join(plot_dir,
                                       '{}_distortion_residualrms_per_frame.pdf'.format(name_seed))
            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
        # 1/0

########################################################################################
########################################################################################


def polynomialModel(Alm, C):
    return np.array(C.T * np.mat(Alm).T).flatten()


def weightedPolynomialResiduals(Alm, data):
    # ATTENTION, these residuals are weighted by the uncertainties.

    LHS, xx, yy, ex, ey, dfexpr = data
    k = np.int(np.sqrt(8 * len(Alm) + 1) - 1)
    C, tmp = bivariate_polynomial(xx, yy, k, verbose=0)
    P = sympy.symbols('p0:%d' % len(Alm))
    replacements = (','.join(['("p%d",Alm[%d])' % (d, d) for d in range(len(Alm))]))
    dffunc = dfexpr.subs(eval('[' + replacements + ']'))

    if ((x in dffunc.free_symbols) & (y not in dffunc.free_symbols)):
        wfunc = lambdify((x), dffunc, 'numpy')  # returns a numpy-ready function
        w = ey ** 2 + ex ** 2 * wfunc(xx).flatten() ** 2
    elif ((y in dffunc.free_symbols) & (x not in dffunc.free_symbols)):
        wfunc = lambdify((y), dffunc, 'numpy')
        w = ey ** 2 + ex ** 2 * wfunc(yy).flatten() ** 2
    elif ((x in dffunc.free_symbols) & (y in dffunc.free_symbols)):
        wfunc = lambdify((x, y), dffunc, 'numpy')
        w = ey ** 2 + ex ** 2 * wfunc(xx, yy).flatten() ** 2
    else:
        w = ey ** 2 + ex ** 2 * np.float(dffunc[0])

    # wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
    wi = np.sqrt(1. / w)
    omc = (np.array(LHS).flatten() - polynomialModel(Alm, C))
    d = wi * omc
    #     print('RMS %f' % np.std(omc))
    return d


def polynomialResiduals(Alm, data):
    # O-C residuals (unweighted)
    LHS, xx, yy, ex, ey, dfexpr = data
    k = np.int(np.sqrt(8 * len(Alm) + 1) - 1)
    C, tmp = bivariate_polynomial(xx, yy, k, verbose=0)
    omc = (np.array(LHS).flatten() - polynomialModel(Alm, C))
    return omc


def fitPolynomialWithUncertaintyInXandY(LHS, ximinusx0, yiminusy0, uncertainty_X, uncertainty_LHS, initialParameters,
                                        axis='x', verbose=0):
    Nalm = len(initialParameters)
    k = np.int(np.sqrt(8 * Nalm + 1) - 1)
    P = sympy.symbols('p0:%d' % Nalm)
    Cs, polynomialTermOrder = bivariate_polynomial(x, y, k, verbose=0)
    polynomial = Matrix(Cs.T) * Matrix(P)

    if axis == 'x':
        dfexpr = sympy.diff(polynomial, x)
    elif axis == 'y':
        dfexpr = sympy.diff(polynomial, y)

    data = (LHS, ximinusx0, yiminusy0, uncertainty_X, uncertainty_LHS, dfexpr)
    #     print(polynomialResiduals(initialParameters,data))

    fitobj = kmpfit.Fitter(residuals=weightedPolynomialResiduals, data=data, ftol=1e-19, gtol=1e-19, xtol=1e-19,
                           covtol=1e-19)
    fitobj.fit(params0=initialParameters)

    #     if fitobj.xerror[0] == 0:
    #         verbose=1

    if verbose:
        print(Matrix(P))
        print(Matrix(Cs.T) * Matrix(P))
        print(dfexpr)
        print("======== Results kmpfit: weights for both coordinates ========= axis=%s " % axis)
        print("Fitted parameters:      ", fitobj.params)
        print("Covariance errors:      ", fitobj.xerror)
        print("Standard errors         ", fitobj.stderr)
        print("Chi^2 min:              ", fitobj.chi2_min)
        print("Reduced Chi^2:          ", fitobj.rchi2_min)
        print("Iterations:             ", fitobj.niter)
        print('xtol\t', fitobj.xtol)
        print('gtol\t', fitobj.gtol)
        print('ftol\t', fitobj.ftol)
        print('covtol\t', fitobj.covtol)
        print('status\t', fitobj.status)
        print('parinfo\t', fitobj.parinfo)
        print('params0\t', fitobj.params0)
        print('npegged\t', fitobj.npegged)
        print('orignorm\t', fitobj.orignorm)
        print('nfev\t', fitobj.nfev)

    #     fitobj.nfree is the number of free parameters, NOT degrees of freedom
    Nmes = LHS.shape[1]
    Nparam = Nalm
    omc = polynomialResiduals(fitobj.params, data)
    Nfree = Nmes - Nparam

    # fake a linearfit object
    res = linearfit.LinearFit(np.mat(1), np.mat(1), np.mat(1))
    res.p = fitobj.params
    res.p_normalised_uncertainty = fitobj.xerror
    res.p_formal_uncertainty = fitobj.stderr
    res.p_formal_covariance_matrix = fitobj.covar
    res.residuals = omc
    res.fit = None
    res.chi2 = fitobj.chi2_min
    res.n_freedom = Nfree

    return res


########################################################################################
########################################################################################


def getLazAstrometryCoefficientsFlexible(mp_input, k, reference_frame_number, targetIndex=None, referencePoint=None,
                                         usePositionUncertainties=0, maxdeg=None, useReducedCoordinates=1,
                                         considerUncertaintiesInXandY=0, verbose=0, masked_stars=None):
    """Perform a 2D polynomial fit between the reference_frame_number and all other 'frames' contained in mp.p

    The corresponding polynomial coefficients returned in the lazAstrometryCoefficients object transform
    from the reference_frame_number frame to all the other frames


    :param mp: instance of multiEpochAstrometry
    :param k: mode
    :param reference_frame_number:
    :param targetIndex:
    :param referencePoint:
    :param usePositionUncertainties:
    :param maxdeg:
    :param useReducedCoordinates:
    :param considerUncertaintiesInXandY:
    :param verbose:
    :param masked_stars:
    :return:
    """

    # to avoid unintended side effects
    mp = copy.deepcopy(mp_input)

    #     masked_stars is an array of indices in p that will not be considered in the fit
    if masked_stars is not None:
        if verbose:
            print('Masked %d stars in fit (of %d total)' % (len(masked_stars), mp.p.shape[1]))
            # print(masked_stars)
        mp.p = np.delete(mp.p, masked_stars, axis=1)

    # array holding the data
    p = mp.p

    #     Nframes corresponds to the number of catalogs
    Nframes, Nstars, Nfields = p.shape

    # ix = np.where(mp.colNames == 'x')[0][0]
    # iy = np.where(mp.colNames == 'y')[0][0]
    # isx = np.where(mp.colNames == 'sigma_x')[0][0]
    # isy = np.where(mp.colNames == 'sigma_y')[0][0]

    cal_names_list = mp.colNames.tolist()
    ix = cal_names_list.index('x')
    iy = cal_names_list.index('y')
    isx = cal_names_list.index('sigma_x')
    isy = cal_names_list.index('sigma_y')
    # iy = np.where(mp.colNames == 'y')[0][0]
    # isx = np.where(mp.colNames == 'sigma_x')[0][0]
    # isy = np.where(mp.colNames == 'sigma_y')[0][0]


    # if we do not consider uncertainties in dependent and independent variables explicitly, add them quadratically so they are approximately accounted for in the linear fit
    if considerUncertaintiesInXandY == 0:
        not_reference_frame_number = np.setdiff1d(np.arange(p.shape[0]), np.array([reference_frame_number]))
        p[not_reference_frame_number, :, isx] = np.sqrt(
            p[not_reference_frame_number, :, isx] ** 2 + p[reference_frame_number, :, isx] ** 2)
        p[not_reference_frame_number, :, isy] = np.sqrt(
            p[not_reference_frame_number, :, isy] ** 2 + p[reference_frame_number, :, isy] ** 2)

    # compute coordinate differences between refStars and reference point or target (input to get coefficients of Phi-function)
    p_dif = np.copy(p)
    p_dif = p_dif.swapaxes(0, 2).swapaxes(0, 1)

    if targetIndex is not None:
        referencePositionX = p_dif[targetIndex, ix, :]
        referencePositionY = p_dif[targetIndex, iy, :]
    elif referencePoint is not None:
        #       if len(reference_point) > 1referencePoint.shape[0] != Nframes#           sys.error('')#         1/0
        referencePositionX = referencePoint[:, 0]
        referencePositionY = referencePoint[:, 1]

    p_dif[:, ix, :] = (p_dif[:, ix, :] - referencePositionX)
    p_dif[:, iy, :] = (p_dif[:, iy, :] - referencePositionY)
    p_dif = p_dif.swapaxes(0, 2).swapaxes(1, 2)

    # compute reduced coordinates, i.e. coordinate differences relative to reference frame
    #     p_red = np.copy(p)
    p_red = np.copy(p_dif)  # change introduced 2016-12-28 JSA

    if useReducedCoordinates:
        #         p_red[:,:,[ix,iy]] = p[    :,:,[ix,iy]] - np.expand_dims(p[    reference_frame_number,:,[ix,iy]],axis=2)
        #         p_red[:,:,[ix,iy]] = p_dif[:,:,[ix,iy]] - np.expand_dims(p_dif[reference_frame_number,:,[ix,iy]],axis=2)
        p_red[:, :, ix] = p[:, :, ix] - p[reference_frame_number, :, ix]
        p_red[:, :, iy] = p[:, :, iy] - p[reference_frame_number, :, iy]

    # to hold uncertainties in p
    s_p = np.zeros(p.shape)
    s_p[:, :, [ix, iy]] = p[:, :, [isx, isy]]

    # to hold uncertainties in p_red
    s_p_red = np.zeros(p.shape)
    if useReducedCoordinates:
        #         s_p_red[:,:,[ix,iy]] = np.sqrt( s_p[:,:,[ix,iy]]**2 + np.expand_dims(s_p[reference_frame_number,:,[ix,iy]]**2,axis=2) )
        s_p_red[:, :, ix] = np.sqrt(s_p[:, :, ix] ** 2 + s_p[reference_frame_number, :, ix] ** 2)
        s_p_red[:, :, iy] = np.sqrt(s_p[:, :, iy] ** 2 + s_p[reference_frame_number, :, iy] ** 2)
    else:
        s_p_red[:, :, [ix, iy]] = s_p[:, :, [ix, iy]]

    # number of polynomial coefficients  per axis (k=4 => Nalm = 3)
    Nalm = np.int(k * (k + 2) / 8)
    if Nalm > Nstars:
        raise RuntimeError('NOT ENOUGH REFERENCE STARS: {} stars, {} free parameters per axis'.format(Nstars, Nalm))

        # arrays to hold polynomial coefficients and their uncertainties
    Alm = np.zeros((Nframes, Nalm * 2))           # holds parameters in X and Y
    s_Alm_normal = np.zeros((Nframes, Nalm * 2))  # holds renormalised uncertainties
    s_Alm_formal = np.zeros((Nframes, Nalm * 2))  # holds formal uncertainties
    rms = np.zeros((Nframes, 2))                  # holds rms parameters in X and Y

    # object arrays to hold linearfit results
    sresx = np.ndarray((Nframes,), dtype=np.object)
    sresy = np.ndarray((Nframes,), dtype=np.object)

    #     solve Thesis Eq.4.18
    #     for X for frame m = 1, exlude target by setting weigth to zero

    # basis for polynomial terms (x-x_0 always computed in reference frame), this is the link between different frames
    ximinusx0 = p_dif[reference_frame_number, :, ix]
    yiminusy0 = p_dif[reference_frame_number, :, iy]

    # uncertainties = uncertainties in p because reference point is exact
    s_ximinusx0 = s_p[reference_frame_number, :, ix]
    s_yiminusy0 = s_p[reference_frame_number, :, iy]

    # get all the polynomial terms up to power k/2 - 1
    C, polynomialTermOrder = bivariate_polynomial(ximinusx0, yiminusy0, k, verbose=verbose)

    data = {}

    for m in range(Nframes):

        # left hand side of matrix equation (Thesis Eq.4.18)
        LHSx = np.mat(p_red[m, :, ix])  # left hand side x
        LHSy = np.mat(p_red[m, :, iy])  # left hand side y

        if usePositionUncertainties == 1:
            uncertainty_x = s_p_red[m, :, ix]
            uncertainty_y = s_p_red[m, :, iy]
        else:
            onesvec = np.ones(Nstars)
            uncertainty_x = onesvec
            uncertainty_y = onesvec

        # if (np.any(uncertainty_x == 0)) | (np.any(uncertainty_y == 0)):
        #     raise RuntimeError('Uncertainty is zero, weight is infinity')


        # set weights to use in linear fit
        if 'artificial' in mp.colNames:
            weight_x = np.zeros(len(uncertainty_x))
            weight_y = np.zeros(len(uncertainty_y))

            artificial_star = mp.p[m, :, mp.colNames.tolist().index('artificial')]
            measured_star_index = artificial_star==0
            artificial_star_index = artificial_star==1

            weight_x[artificial_star_index] = 0
            weight_y[artificial_star_index] = 0
            weight_x[measured_star_index] = 1. / np.power(uncertainty_x[measured_star_index], 2)
            weight_y[measured_star_index] = 1. / np.power(uncertainty_y[measured_star_index], 2)
        else:
            weight_x = 1. / np.power(uncertainty_x, 2)
            weight_y = 1. / np.power(uncertainty_y, 2)

        # if a target is used as reference point, set its weight to zero (target is first element in mp.p)
        if targetIndex is not None:
            weight_x[targetIndex] = 0
            weight_y[targetIndex] = 0

        if (np.any(np.isinf(weight_x)) | np.any(np.isinf(weight_y))):
            raise RuntimeError('Weight is infinity')


        # perform the linear fit, yielding the polynomial coefficients
        # the matrix equation has as many rows as reference stars and represents a k/2-1 polynomial
        # linearfit is always needed to provide initial parameters for kmpfit
        resx = linearfit.LinearFit(LHSx, np.diag(weight_x), C)
        resy = linearfit.LinearFit(LHSy, np.diag(weight_y), C)
        try:
            resx.fit()
            resy.fit()
        except np.linalg.LinAlgError:
            1/0

        if considerUncertaintiesInXandY == 1:
            # uncertainty in left hand side of equation (Y in a linear fit)
            uncertainty_LHS_x = uncertainty_x
            uncertainty_LHS_y = uncertainty_y

            uncertainty_polyCoord_x = s_ximinusx0
            uncertainty_polyCoord_y = s_yiminusy0

            initialParameters_x = resx.p  # initialParameters[0] *= 1.1
            initialParameters_y = resy.p  # initialParameters[0] *= 1.1

            # uncertainty in dependent coordinate (X in a linear fit), here this is the uncertainty in ximinusx0 or yiminusy0
            resx = fitPolynomialWithUncertaintyInXandY(LHSx, ximinusx0, yiminusy0, uncertainty_polyCoord_x,
                                                       uncertainty_LHS_x, initialParameters_x, axis='x',
                                                       verbose=verbose)
            resy = fitPolynomialWithUncertaintyInXandY(LHSy, ximinusx0, yiminusy0, uncertainty_polyCoord_y,
                                                       uncertainty_LHS_y, initialParameters_y, axis='y',
                                                       verbose=verbose)

        if 0:
            #         if m!=reference_frame_number:
            1 / 0
            resx.displayResults(precision=10)

            skyAxis = 'x'
            # uncertainty in left hand side of equation (Y in a linear fit)
            uncertainty_LHS = uncertainty_x
            # uncertainty in dependent coordinate (X in a linear fit), here this is the uncertainty in ximinusx0 or yiminusy0
            uncertainty_polyCoord = s_ximinusx0
            initialParameters = resx.p
            resx_sxsy = fitPolynomialWithUncertaintyInXandY(LHSx, ximinusx0, yiminusy0, uncertainty_polyCoord,
                                                            uncertainty_LHS, initialParameters, axis=skyAxis, verbose=0)
            resx_sxsy.displayResults(precision=10)

        if 1:
            # delete input convariance matrix to save space in object
            resx.inverse_covariance_matrix = []
            resy.inverse_covariance_matrix = []

        # store polynomial coefficients
        Alm[m, :] = np.concatenate((resx.p, resy.p)).T

        # uncertainty in polynomial coefficients
        s_Alm_normal[m, :] = np.concatenate((resx.p_normalised_uncertainty, resy.p_normalised_uncertainty)).T
        s_Alm_formal[m, :] = np.concatenate((resx.p_formal_uncertainty, resy.p_formal_uncertainty)).T

        # RMS dispersion of the fit
        # rmsx = resx.residuals.std()
        # rmsy = resy.residuals.std()
        rmsx = resx.residuals[weight_x!=0].std()
        rmsy = resy.residuals[weight_y!=0].std()

        # store the results and the rms
        sresx[m] = resx
        sresy[m] = resy
        rms[m, :] = np.array([rmsx, rmsy])

        data[m] = {}
        data[m]['weight_x'] = weight_x
        data[m]['weight_y'] = weight_y
        data[m]['number_of_stars_with_non_zero_weight'] = len(np.where(weight_x!=0)[0])

    # return object
    return lazAstrometryCoefficients(k=k, p=p, p_dif=p_dif, p_red=p_red, Alm=Alm, s_Alm_normal=s_Alm_normal,
                                     s_Alm_formal=s_Alm_formal, rms=rms, C=C, Nalm=Nalm, resx=sresx, resy=sresy,
                                     s_p_red=s_p_red, polynomialTermOrder=polynomialTermOrder,
                                     referencePoint=referencePoint, useReducedCoordinates=useReducedCoordinates,
                                     refFrameNumber=reference_frame_number, colNames=mp.colNames, data=data)


def computeScaleFromPolyCoeff(pc1, pc2):
    return np.sqrt(pc1 ** 2 + pc2 ** 2)


def computeRotationFromPolyCoeff(pc1, pc2):
    return np.rad2deg(np.arctan2(pc1, pc2))


def computeRotScaleSkewFromPolyCoeff(b, c, e, f):
    xmag = computeScaleFromPolyCoeff(b, e)
    ymag = computeScaleFromPolyCoeff(c, f)
    xrotation = computeRotationFromPolyCoeff(-e, b)
    yrotation = computeRotationFromPolyCoeff(c, f)
    rotation = (xrotation + yrotation) / 2.
    skew = yrotation - xrotation
    #     Jay anderson parameters"
    #     in jay anderson's nomenclature: b=A, c=B, e=C, f=D, see ACS2007-08
    rotation2 = np.rad2deg(np.arctan2(c - e, b + f))
    #     rotation2 = np.arctan2(b+f,c-e)
    relative_scale = np.sqrt(b * f - c * e)
    skew_onaxis = (b - f) / 2.  # difference in x/y scale
    skew_offaxis = (c + e) / 2.  # non-perpincicularity between the axes

    return xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis, skew_offaxis, rotation2


def computePolyCoeffFromRotScaleSkew(xmag, ymag, xrotation, yrotation):
    # http://iraf.net/irafhelp.php?val=immatch.geoxytran&help=Help+Page
    b = xmag * np.cos(xrotation)
    c = ymag * np.sin(yrotation)
    e = -xmag * np.sin(xrotation)
    f = ymag * np.cos(yrotation)

    return b, c, e, f


def compute_rot_scale_skew(lazAC, i=0, scaleFactor=1.):
    """
    same as displayRotScaleSkew but without text output
    """
    xshift = lazAC.Alm[i, 0] * scaleFactor
    yshift = lazAC.Alm[i, lazAC.Nalm] * scaleFactor
    if lazAC.k > 2:
        idx_forXterm = np.where(lazAC.polynomialTermOrder == 'x')[0]
        idx_forYterm = np.where(lazAC.polynomialTermOrder == 'y')[0]

        b = lazAC.Alm[i, idx_forXterm] * scaleFactor  # x term
        c = lazAC.Alm[i, idx_forYterm] * scaleFactor  # y term
        e = lazAC.Alm[i, lazAC.Nalm + idx_forXterm] * scaleFactor  # x term
        f = lazAC.Alm[i, lazAC.Nalm + idx_forYterm] * scaleFactor  # y term

        if 0:
            xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis, skew_offaxis, rotation2 = computeRotScaleSkewFromPolyCoeff(
            b, c, e, f)

        # determine uncertainties assuming Gaussian distributions
        N_mc = 1000
        np.random.seed(0)

        #   using renormalised uncertainties
        b_mc = b + np.random.normal(0., lazAC.s_Alm_normal[i, idx_forXterm] * scaleFactor, N_mc)
        c_mc = c + np.random.normal(0., lazAC.s_Alm_normal[i, idx_forYterm] * scaleFactor, N_mc)
        e_mc = e + np.random.normal(0., lazAC.s_Alm_normal[i, lazAC.Nalm + idx_forXterm] * scaleFactor, N_mc)
        f_mc = f + np.random.normal(0., lazAC.s_Alm_normal[i, lazAC.Nalm + idx_forYterm] * scaleFactor, N_mc)

        #   using formal uncertainties
        b_mcf = b + np.random.normal(0., lazAC.s_Alm_formal[i, idx_forXterm] * scaleFactor, N_mc)
        c_mcf = c + np.random.normal(0., lazAC.s_Alm_formal[i, idx_forYterm] * scaleFactor, N_mc)
        e_mcf = e + np.random.normal(0., lazAC.s_Alm_formal[i, lazAC.Nalm + idx_forXterm] * scaleFactor, N_mc)
        f_mcf = f + np.random.normal(0., lazAC.s_Alm_formal[i, lazAC.Nalm + idx_forYterm] * scaleFactor, N_mc)

        xmag_mc, ymag_mc, xrotation_mc, yrotation_mc, rotation_mc, skew_mc, relative_scale_mc, skew_onaxis_mc, skew_offaxis_mc, rotation2_mc = computeRotScaleSkewFromPolyCoeff(
            b_mc, c_mc, e_mc, f_mc)
        if 0:
            xmag_mcf, ymag_mcf, xrotation_mcf, yrotation_mcf, rotation_mcf, skew_mcf, relative_scale_mcf, skew_onaxis_mcf, skew_offaxis_mcf, rotation2_mcf = computeRotScaleSkewFromPolyCoeff(
            b_mcf, c_mcf, e_mcf, f_mcf)

            deg2arcsec = u.deg.to(u.arcsec)

    dat = np.zeros((12, 2))
    nams = np.array(
        ['Shift in X', 'Shift in Y', 'Rotation in X', 'Rotation in Y', 'Scale in X', 'Scale in Y', 'Rotation', 'Skew',
         'Global Rotation', 'Global Scale', 'On-axis Skew', 'Off-axis Skew'])
    units = ['', '', 'deg', 'deg', '', '', 'deg', 'deg', 'deg', '', '', '']

    dat[0, :] = [xshift, lazAC.s_Alm_normal[i, 0]]
    dat[1, :] = [yshift, lazAC.s_Alm_normal[i, lazAC.Nalm]]
    if lazAC.k > 2:
        dat[2, :] = [np.mean(xrotation_mc), np.std(xrotation_mc)]
        dat[3, :] = [np.mean(yrotation_mc), np.std(yrotation_mc)]
        dat[4, :] = [np.mean(xmag_mc), np.std(xmag_mc)]
        dat[5, :] = [np.mean(ymag_mc), np.std(ymag_mc)]
        dat[6, :] = [np.mean(rotation_mc), np.std(rotation_mc)]
        dat[7, :] = [np.mean(skew_mc), np.std(skew_mc)]
        dat[8, :] = [np.mean(rotation2_mc), np.std(rotation2_mc)]
        dat[9, :] = [np.mean(relative_scale_mc), np.std(relative_scale_mc)]
        dat[10, :] = [np.mean(skew_onaxis_mc), np.std(skew_onaxis_mc)]
        dat[11, :] = [np.mean(skew_offaxis_mc), np.std(skew_offaxis_mc)]

    human_readable_solution_parameters = {'values': dat, 'names': nams, 'units': units}

    return human_readable_solution_parameters


def displayRotScaleSkew(lazAC, i=0, scaleFactor=1., nformat='f'):
    xshift = lazAC.Alm[i, 0] * scaleFactor
    yshift = lazAC.Alm[i, lazAC.Nalm] * scaleFactor
    idx_forXterm = np.where(lazAC.polynomialTermOrder == 'x')[0]
    idx_forYterm = np.where(lazAC.polynomialTermOrder == 'y')[0]

    b = lazAC.Alm[i, idx_forXterm] * scaleFactor  # x term
    c = lazAC.Alm[i, idx_forYterm] * scaleFactor  # y term
    e = lazAC.Alm[i, lazAC.Nalm + idx_forXterm] * scaleFactor  # x term
    f = lazAC.Alm[i, lazAC.Nalm + idx_forYterm] * scaleFactor  # y term

    xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis, skew_offaxis, rotation2 = computeRotScaleSkewFromPolyCoeff(
        b, c, e, f)

    # determine uncertainties assuming Gaussian distributions
    N_mc = 1000
    np.random.seed(0)

    #   using renormalised uncertainties
    b_mc = b + np.random.normal(0., lazAC.s_Alm_normal[i, idx_forXterm] * scaleFactor, N_mc)
    c_mc = c + np.random.normal(0., lazAC.s_Alm_normal[i, idx_forYterm] * scaleFactor, N_mc)
    e_mc = e + np.random.normal(0., lazAC.s_Alm_normal[i, lazAC.Nalm + idx_forXterm] * scaleFactor, N_mc)
    f_mc = f + np.random.normal(0., lazAC.s_Alm_normal[i, lazAC.Nalm + idx_forYterm] * scaleFactor, N_mc)

    #   using formal uncertainties
    b_mcf = b + np.random.normal(0., lazAC.s_Alm_formal[i, idx_forXterm] * scaleFactor, N_mc)
    c_mcf = c + np.random.normal(0., lazAC.s_Alm_formal[i, idx_forYterm] * scaleFactor, N_mc)
    e_mcf = e + np.random.normal(0., lazAC.s_Alm_formal[i, lazAC.Nalm + idx_forXterm] * scaleFactor, N_mc)
    f_mcf = f + np.random.normal(0., lazAC.s_Alm_formal[i, lazAC.Nalm + idx_forYterm] * scaleFactor, N_mc)

    #     b_mc = b + scipy.randn(N_mc) * lazAC.s_Alm[i,           idx_forXterm]
    #     c_mc = c + scipy.randn(N_mc) * lazAC.s_Alm[i,           idx_forYterm]
    #     e_mc = e + scipy.randn(N_mc) * lazAC.s_Alm[i,lazAC.Nalm+idx_forXterm]
    #     f_mc = f + scipy.randn(N_mc) * lazAC.s_Alm[i,lazAC.Nalm+idx_forYterm]
    xmag_mc, ymag_mc, xrotation_mc, yrotation_mc, rotation_mc, skew_mc, relative_scale_mc, skew_onaxis_mc, skew_offaxis_mc, rotation2_mc = computeRotScaleSkewFromPolyCoeff(
        b_mc, c_mc, e_mc, f_mc)
    xmag_mcf, ymag_mcf, xrotation_mcf, yrotation_mcf, rotation_mcf, skew_mcf, relative_scale_mcf, skew_onaxis_mcf, skew_offaxis_mcf, rotation2_mcf = computeRotScaleSkewFromPolyCoeff(
        b_mcf, c_mcf, e_mcf, f_mcf)

    #     xmag_mc  *= scaleFactor
    #     ymag_mc  *= scaleFactor
    #     xmag_mcf *= scaleFactor
    #     ymag_mcf *= scaleFactor

    deg2arcsec = u.deg.to(u.arcsec)
    print('Classical quantitites of the polynomial:')
    if nformat == 'f':
        #         print('Shift    in X / Y  =  %3.6f +/- %3.6f (formal %3.6f)      /  %3.6f +/- %3.6f (formal %3.6f)    ' % (xshift*scaleFactor,lazAC.s_Alm_normal[i,0]*scaleFactor,lazAC.s_Alm_formal[i,0]*scaleFactor,yshift*scaleFactor,lazAC.s_Alm_normal[i,lazAC.Nalm]*scaleFactor,lazAC.s_Alm_formal[i,lazAC.Nalm]*scaleFactor) )
        print('Shift    in X / Y  =  %3.6f +/- %3.6f (formal %3.6f)      /  %3.6f +/- %3.6f (formal %3.6f)    ' % (
        xshift, lazAC.s_Alm_normal[i, 0], lazAC.s_Alm_formal[i, 0], yshift, lazAC.s_Alm_normal[i, lazAC.Nalm],
        lazAC.s_Alm_formal[i, lazAC.Nalm]))
        print('Rotation in X / Y  =  %3.6f +/- %3.6f (formal %3.6f) deg  /  %3.6f +/- %3.6f (formal %3.6f) deg' % (
        np.mean(xrotation_mc), np.std(xrotation_mc), np.std(xrotation_mcf), np.mean(yrotation_mc), np.std(yrotation_mc),
        np.std(yrotation_mcf)))
        print('Scale    in X / Y  =  %3.6f +/- %3.6f (formal %3.6f)      /  %3.6f +/- %3.6f (formal %3.6f)    ' % (
        np.mean(xmag_mc), np.std(xmag_mc), np.std(xmag_mcf), np.mean(ymag_mc), np.std(ymag_mc), np.std(ymag_mcf)))
        print('Rotation / Skew    =  %3.6f +/- %3.6f (formal %3.6f) deg  /  %3.6f +/- %3.6f (formal %3.6f) deg' % (
        np.mean(rotation_mc), np.std(rotation_mc), np.std(rotation_mcf), np.mean(skew_mc), np.std(skew_mc),
        np.std(skew_mcf)))
        print('Rotation2          =  %3.6f +/- %3.6f (formal %3.6f) deg' % (
        np.mean(rotation2_mc), np.std(rotation2_mc), np.std(rotation2_mcf)))
        print('Relative scale     =  %3.6f +/- %3.6f (formal %3.6f) ' % (
        np.mean(relative_scale_mc), np.std(relative_scale_mc), np.std(relative_scale_mcf)))
        print(' On-axis Skew      =  %3.6f +/- %3.6f (formal %3.6f) ' % (
        np.mean(skew_onaxis_mc), np.std(skew_onaxis_mc), np.std(skew_onaxis_mcf)))
        print('Off-axis Skew      =  %3.6f +/- %3.6f (formal %3.6f) ' % (
        np.mean(skew_offaxis_mc), np.std(skew_offaxis_mc), np.std(skew_offaxis_mcf)))
    elif nformat == 'e':
        print('Shift    in X / Y  =  %3.6e +/- %3.6e (formal %3.6e)      /  %3.6e +/- %3.6e (formal %3.6e)    ' % (
        xshift, lazAC.s_Alm_normal[i, 0], lazAC.s_Alm_formal[i, 0], yshift, lazAC.s_Alm_normal[i, lazAC.Nalm],
        lazAC.s_Alm_formal[i, lazAC.Nalm]))
        print('Rotation in X / Y  =  %3.6e +/- %3.6e (formal %3.6e) deg  /  %3.6e +/- %3.6e (formal %3.6e) deg' % (
        np.mean(xrotation_mc), np.std(xrotation_mc), np.std(xrotation_mcf), np.mean(yrotation_mc), np.std(yrotation_mc),
        np.std(yrotation_mcf)))
        print('Scale    in X / Y  =  %3.6e +/- %3.6e (formal %3.6e)      /  %3.6e +/- %3.6e (formal %3.6e)    ' % (
        np.mean(xmag_mc), np.std(xmag_mc), np.std(xmag_mcf), np.mean(ymag_mc), np.std(ymag_mc), np.std(ymag_mcf)))
        print('Rotation / Skew    =  %3.6e +/- %3.6e (formal %3.6e) deg  /  %3.6e +/- %3.6e (formal %3.6e) deg' % (
        np.mean(rotation_mc), np.std(rotation_mc), np.std(rotation_mcf), np.mean(skew_mc), np.std(skew_mc),
        np.std(skew_mcf)))
        print('Rotation2          =  %3.6e +/- %3.6e (formal %3.6e) deg' % (
        np.mean(rotation2_mc), np.std(rotation2_mc), np.std(rotation2_mcf)))
        print('Relative scale     =  %3.6e +/- %3.6e (formal %3.6e) ' % (
            np.mean(relative_scale_mc), np.std(relative_scale_mc), np.std(relative_scale_mcf)))
        print(' On-axis Skew      =  %3.6e +/- %3.6e (formal %3.6e) ' % (
        np.mean(skew_onaxis_mc), np.std(skew_onaxis_mc), np.std(skew_onaxis_mcf)))
        print('Off-axis Skew      =  %3.6e +/- %3.6e (formal %3.6e) ' % (
        np.mean(skew_offaxis_mc), np.std(skew_offaxis_mc), np.std(skew_offaxis_mcf)))

    print('Rotation           =  %3.4f +/- %3.4f (formal %3.4f) arcsec' % (
    np.mean(rotation_mc) * deg2arcsec, np.std(rotation_mc) * deg2arcsec, np.std(rotation_mcf) * deg2arcsec))
    print('Rotation2          =  %3.4f +/- %3.4f (formal %3.4f) arcsec' % (
    np.mean(rotation2_mc) * deg2arcsec, np.std(rotation2_mc) * deg2arcsec, np.std(rotation2_mcf) * deg2arcsec))
    #     print('Off-axis Skew      =  %3.4f +/- %3.4f (formal %3.2f) arcsec' % (np.mean(skew_offaxis_mc)*deg2arcsec,np.std(skew_offaxis_mc)*deg2arcsec,np.std(skew_offaxis_mcf)*deg2arcsec))

    dat = np.zeros((12, 2))
    nams = np.array(
        ['Shift in X', 'Shift in Y', 'Rotation in X', 'Rotation in Y', 'Scale in X', 'Scale in Y', 'Rotation', 'Skew',
         'Global Rotation', 'Global Scale', 'On-axis Skew', 'Off-axis Skew'])
    units = ['', '', 'deg', 'deg', '', '', 'deg', 'deg', 'deg', '', '', '']

    dat[0, :] = [xshift, lazAC.s_Alm_normal[i, 0]]
    dat[1, :] = [yshift, lazAC.s_Alm_normal[i, lazAC.Nalm]]
    dat[2, :] = [np.mean(xrotation_mc), np.std(xrotation_mc)]
    dat[3, :] = [np.mean(yrotation_mc), np.std(yrotation_mc)]
    dat[4, :] = [np.mean(xmag_mc), np.std(xmag_mc)]
    dat[5, :] = [np.mean(ymag_mc), np.std(ymag_mc)]
    dat[6, :] = [np.mean(rotation_mc), np.std(rotation_mc)]
    dat[7, :] = [np.mean(skew_mc), np.std(skew_mc)]
    dat[8, :] = [np.mean(rotation2_mc), np.std(rotation2_mc)]
    dat[9, :] = [np.mean(relative_scale_mc), np.std(relative_scale_mc)]
    dat[10, :] = [np.mean(skew_onaxis_mc), np.std(skew_onaxis_mc)]
    dat[11, :] = [np.mean(skew_offaxis_mc), np.std(skew_offaxis_mc)]
    for j, nam in enumerate(nams):
        print('%s,%3.7f,%3.7f,%s' % (nam, dat[j, 0], dat[j, 1], units[j]))
    for j, nam in enumerate(nams):
        print('%s,%2.2e,%2.2e,%s' % (nam, dat[j, 0], dat[j, 1], units[j]))

    human_readable_solution_parameters = {'values': dat, 'names': nams, 'units': units}

    return human_readable_solution_parameters




def fitDistortion(mp, k, reference_frame_number=0, evaluation_frame_number=1, targetId=None,
                  use_position_uncertainties=1, reference_point=None, debug=0, use_reduced_coordinates=0,
                  consider_uncertainties_in_XY=0, verbose=0, index_masked_stars=None):
    """
    Fit geometric distortion using 2D polynomial of mode k between catalogs/frames contained in multiEpochAstrometry object mp
    written 2016-12 J. Sahlmann, AURA/STScI



    Parameters
    ----------
    mp : multiEpochAstrometry object
        object containing the astrometric data in the numpy array mp.p
    k : int
        mode of the 2D polynomial of order k/2-1

    """

    targetIndex = None

    if debug == 1:
        p = mp.p
        x = p[:, :, 2]
        y = p[:, :, 3]
        # display vector plot
        X, Y = x[0, :], y[0, :]
        U, V = x[1, :] - x[0, :], y[1, :] - y[0, :]

        pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        pl.clf()
        Q = pl.quiver(X, Y, U, V, angles='xy')
        #       ax = pl.gca()
        #       ax.invert_xaxis()
        #       ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #       ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        pl.title('Difference between catalog positions')
        pl.show()

        fig = pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        pl.clf()
        pl.hist(U, 50, color='b', label='X')
        pl.hist(V, 50, color='r', alpha=0.5, label='Y')
        pl.xlabel('Coordinate Difference in X and Y ')
        pl.legend(loc='best')
        fig.tight_layout(h_pad=0.0)
        pl.show()

        pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        pl.clf()
        pl.plot(x[0, :], y[0, :], 'bo')
        pl.plot(x[1, :], y[1, :], 'ro')
        ax = pl.gca()
        ax.invert_xaxis()
        pl.title('Catalog positions')
        pl.show()

    if (k == 2) & verbose:
        # has to use reduced coordinates, otherwise yields senseless results
        use_reduced_coordinates = 1
        print('Mode k = {0:d}, forced to use reduced coordinates!'.format(k))

    if targetId is not None:
        # data for reference stars only
        s = getRefStarAstrometricData(mp, targetId)
        #     reconstruct p-array from good reference stars and target
        mp, targetIndex = getCleanedAstrometricData(mp, s, targetId)  # target is now first element
    elif reference_point is not None:
        targetIndex = None
        pass
    else:
        os.error('No reference point defined!')

    lazAC = getLazAstrometryCoefficientsFlexible(mp, k, reference_frame_number, referencePoint=reference_point,
                                                 targetIndex=targetIndex,
                                                 usePositionUncertainties=use_position_uncertainties,
                                                 useReducedCoordinates=use_reduced_coordinates,
                                                 considerUncertaintiesInXandY=consider_uncertainties_in_XY,
                                                 verbose=verbose, masked_stars=index_masked_stars)

    return lazAC


def fit_distortion_general(mp, k,
                           eliminate_omc_outliers_iteratively=0,
                           outlier_rejection_level_sigma=5.,
                           reference_frame_number=0,
                           evaluation_frame_number=1,
                           reference_point=None,
                           use_position_uncertainties=1,
                           use_reduced_coordinates=0,
                           consider_uncertainties_in_XY=0,
                           verbose=False,
                           index_masked_stars=None):
    """

    :param mp:
    :param k:
    :param reference_frame_number:
    :param evaluation_frame_number:
    :param reference_point:
    :param use_position_uncertainties:
    :param use_reduced_coordinates:
    :param consider_uncertainties_in_XY:
    :param verbose:
    :return:
    """

    if eliminate_omc_outliers_iteratively:
        # loop while eliminating outliers
        mp_orig = copy.deepcopy(mp)
        # index_masked_stars = None
        number_of_outliers = -1
        iteration = 0
        index_in_orig_mp = np.arange(mp_orig.p.shape[1])
        while number_of_outliers != 0:
            mp = copy.deepcopy(mp_orig)
            lazAC = fitDistortion(mp, k, reference_frame_number=reference_frame_number,
                                  evaluation_frame_number=evaluation_frame_number,
                                  reference_point=reference_point,
                                  use_position_uncertainties=use_position_uncertainties,
                                  use_reduced_coordinates=use_reduced_coordinates,
                                  consider_uncertainties_in_XY=consider_uncertainties_in_XY,
                                  verbose=verbose,
                                  index_masked_stars=index_masked_stars)

            residuals_x = lazAC.resx[evaluation_frame_number].residuals
            residuals_y = lazAC.resy[evaluation_frame_number].residuals
            if verbose:
                print(
                    'Iteration %d Number of %3.1f sigma outlier residuals %d, residual RMS %3.3f in X and %3.3f in Y' % (
                        iteration, outlier_rejection_level_sigma, number_of_outliers, np.std(residuals_x),
                        np.std(residuals_y)))
            # array of indices of outliers
            index_in_masked_mp = np.where(
                (np.abs(residuals_x) > outlier_rejection_level_sigma * np.std(residuals_x)) | (
                    np.abs(residuals_y) > outlier_rejection_level_sigma * np.std(residuals_y)))[0]
            number_of_outliers = len(index_in_masked_mp)
            if number_of_outliers != 0:
                temp_masked_stars = index_in_orig_mp[index_in_masked_mp]
                if iteration == 0:
                    index_masked_stars = temp_masked_stars
                else:
                    index_masked_stars = np.hstack((index_masked_stars, temp_masked_stars))
                index_in_orig_mp = np.delete(index_in_orig_mp, index_in_masked_mp)
                iteration += 1
                if verbose:
                    print(
                        'Sigma clipping of residual outliers converged after %d iterations. Final number of outliers is %d' % (
                        iteration, len(index_masked_stars)))



    else:
        lazAC = fitDistortion(mp, k, reference_frame_number=reference_frame_number,
                              evaluation_frame_number=evaluation_frame_number,
                              reference_point=reference_point,
                              use_position_uncertainties=use_position_uncertainties,
                              use_reduced_coordinates=use_reduced_coordinates,
                              consider_uncertainties_in_XY=consider_uncertainties_in_XY,
                              verbose=verbose,
                              index_masked_stars=index_masked_stars)

    return lazAC, index_masked_stars


def rotate(origin, px, py, angle_rad):
    """
    Rotate a point clockwise by a given angle around a given origin.
    If RA,Dec = px,py, then this corresponds to a rotation of angle_rad from North to east

    The angle should be given in radians.
    """
    ox, oy = origin
    #     px, py = point

    qx = ox + np.cos(-angle_rad) * (px - ox) - np.sin(-angle_rad) * (py - oy)
    qy = oy + np.sin(-angle_rad) * (px - ox) + np.cos(-angle_rad) * (py - oy)
    return qx, qy


# import warnings
# from astropy.utils.exceptions import AstropyWarning
# import pygacs.public.publicAccessTools as pgp
#
#
# def return_gacs_query_as_table(query_string,output_file_seed,overwrite=0):
#     """
#     run query on Gaia archive (GACS) and return result as astropy table
#     """
#     if (overwrite == 1) | (not os.path.isfile(output_file_seed+'.vot')):
#         pgp.retrieveQueryResult(query_string,output_file_seed+'.vot')
#         d = Table.read(outputFileSeed+'.vot',format='votable')
#         problematic_columns = ['tycho2_id','phot_variable_flag']
#         for colname in problematic_columns:
#             if colname in d.colnames:
#                 tmp = np.array(d[colname]).astype(np.str)
#                 d.remove_column(colname)
#                 d[colname] = tmp
#         d.write(output_file_seed+'.fits',overwrite=overwrite)
#     else:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', AstropyWarning)
#             d = Table.read(output_file_seed+'.fits')
#             print('Retrieved %d sources from Gaia catalog in this field'%(len(d)))
#
#     return d


def testDistortionFit():
    print('*' * 30, ' TEST DISTORTION FIT ', '*' * 30)

    # center of test field in RA and Dec
    #     RA0_deg = 0.
    #     DE0_deg = 0.

    RA0_deg = 80.
    DE0_deg = -70.

    # rotation angle between the two catalogs
    angle_deg = -4.
    # origin of rotation
    #     origin = (0.,0.)
    origin = (RA0_deg, DE0_deg)

    if 1 == 1:  # coordinates on grid
        Ngrid = 10
        fieldsize = 2. * u.arcmin
        ra_max_deg = fieldsize.to(u.deg).value / 2.
        de_max_deg = fieldsize.to(u.deg).value / 2.

        ra_deg = np.linspace(RA0_deg - ra_max_deg, RA0_deg + ra_max_deg, Ngrid)
        de_deg = np.linspace(DE0_deg - de_max_deg, DE0_deg + de_max_deg, Ngrid)
        ra_deg_mesh, de_deg_mesh = np.meshgrid(ra_deg, de_deg)
        ra_deg_all = ra_deg_mesh.flatten()
        de_deg_all = de_deg_mesh.flatten()


    else:  # coordinates on concentric circles
        def PointsInCircum(r, n=10):
            return np.array([[np.cos(2 * np.pi / n * x) * r, np.sin(2 * np.pi / n * x) * r] for x in xrange(0, n + 1)])

        radius_max_deg = 0.001
        Ncircles = 5
        NpointPerCircle = 10
        #         points = np.array([])#np.zeros(NpointPerCircle*Ncircles,2)
        for i in range(Ncircles):
            r = radius_max_deg / (i + 1)
            if i == 0:
                points = PointsInCircum(r, n=NpointPerCircle)
            else:
                points = np.vstack((points, PointsInCircum(r, n=NpointPerCircle)))
        ra_deg_all = points[:, 0]
        de_deg_all = points[:, 1]

    # prepare regular coordinate grid
    T = Table((ra_deg_all, de_deg_all), names=('ra', 'dec'))

    # prepare rotated coordinate grid table
    ra2, dec2 = rotate(origin, T['ra'], T['dec'], np.deg2rad(angle_deg))
    T2 = Table((ra2, dec2), names=('ra', 'dec'))
    T2['Id'] = range(len(T2))

    Nstars = len(T)

    #     introduce measurement noise
    if 1 == 1:
        astrometric_uncertainty = 1.0 * u.milliarcsecond
        np.random.seed(0)
        T['ra'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value, Nstars)
        np.random.seed(1)
        T['dec'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value, Nstars)
        np.random.seed(2)
        T2['ra'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value, Nstars)
        np.random.seed(3)
        T2['dec'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value, Nstars)

    testSkyPixelMatch = 0
    if testSkyPixelMatch:
        # generate corresponding pixel positions

        x = np.arange(Ngrid)
        y = np.arange(Ngrid)
        x_mesh, y_mesh = np.meshgrid(x, y)
        x_all = x_mesh.flatten()
        y_all = y_mesh.flatten()
        #         T = Table((x_all,y_all),names=('x','y'))
        T = Table((x_all, y_all), names=('ra', 'dec'))

    T['Id'] = range(len(T))

    if 0 == 1:
        pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        pl.clf()
        pl.plot(T['ra'], T['dec'], 'bo')
        pl.plot(T2['ra'], T2['dec'], 'ro')
        ax = pl.gca()
        ax.invert_xaxis()
        pl.xlabel('Right Ascension (deg)')
        pl.ylabel('Declination (deg)')
        pl.title('Difference between catalog positions')
        pl.show()

    # display vector plot
    X, Y = T['ra'], T['dec']
    U, V = T2['ra'] - T['ra'], T2['dec'] - T['dec']

    pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
    pl.clf()
    Q = pl.quiver(X, Y, U, V, angles='xy')
    ax = pl.gca()
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    pl.xlabel('Right Ascension (deg)')
    pl.ylabel('Declination (deg)')
    pl.title('Difference between catalog positions')
    pl.show()

    Nstars = len(T)
    colNames = np.array(['tmp1', 'tmp2', 'x', 'y', 'tmp3', 'tmp4', 'id', 'original_id', 'sigma_x', 'sigma_y'])
    p = np.zeros((2, Nstars, len(colNames)))
    xmatchId = np.array(T['Id'])
    pos_uncertainty_deg = astrometric_uncertainty.to(u.deg).value
    #     pos_uncertainty = 1000*u.milliarcsecond
    #     pos_uncertainty_deg = pos_uncertainty.to(u.deg).value

    x1 = np.array(T['ra'])
    x2 = np.array(T2['ra'])
    y1 = np.array(T['dec'])
    y2 = np.array(T2['dec'])
    id1 = np.array(T['Id'])
    id2 = np.array(T2['Id'])
    for i in range(Nstars):
        p[0, i, :] = [0, 0, x1[i], y1[i], 0, 0, xmatchId[i], id1[i], pos_uncertainty_deg, pos_uncertainty_deg]  # cat 1
        p[1, i, :] = [0, 0, x2[i], y2[i], 0, 0, xmatchId[i], id2[i], pos_uncertainty_deg, pos_uncertainty_deg]  # cat 2

    primaryCat = SkyCoord(ra=T['ra'] * u.degree, dec=T['dec'] * u.degree)

    deg2mas = u.deg.to(u.mas)

    referencePoint = None
    targetId = None

    ###############################
    k = 4
    refFrameNumber = 0
    evaluation_frame_number = 0
    verbose = 0
    verboseFigures = 1
    saveplot = 0
    usePositionUncertainties = 0
    useReducedCoordinates = 0  # use coordinate differences relative to reference frame in the fit

    #     coordType='ra-dec'
    #     coordType='x-y'
    #     coordType='mixed'

    # define the reference point for the differential coordinates. has to have same units as x and y columns, if set to (0,0) the differential coordinates are the same as the coordinates
    # either targetId or reference_point have to be set in the call to fitDistortion
    #     reference_point = [0.,0.]
    #     reference_point = np.array([[5.,5.],[RA0_deg,DE0_deg]])
    referencePoint = np.array([[0., 0.], [RA0_deg, DE0_deg]])
    #     reference_point = (RA0_deg,DE0_deg)
    ###############################

    nameSeed = 'test'
    #     scaleFactor=deg2mas
    scaleFactor = 1.

    outDir = os.path.join(os.path.dirname(__file__), nameSeed)

    # call object constructor
    mp = multiEpochAstrometry(p, colNames)

    # find star close to the field center
    if referencePoint is None:
        centerCat = SkyCoord(ra=RA0_deg * u.degree, dec=DE0_deg * u.degree)
        targIndex = np.where(primaryCat.separation(centerCat) < 30 * u.arcsec)[0][0]
        targetId = xmatchId[targIndex]

    # lazAC = fitDistortion(mp,targetId,k,reference_frame_number,use_position_uncertainties,scaleFactor=scaleFactor, verbose = verbose, verbose_figures = verbose_figures, save_plot = save_plot, outDir = outDir, nameSeed = nameSeed, use_reduced_coordinates=use_reduced_coordinates)
    #     lazAC = fitDistortion(mp,k,reference_frame_number,targetId=targetId,reference_point=reference_point,use_position_uncertainties=use_position_uncertainties,scaleFactor=scaleFactor, verbose = verbose, verbose_figures = verbose_figures, save_plot = save_plot, outDir = outDir, nameSeed = nameSeed, use_reduced_coordinates=use_reduced_coordinates)
    lazAC = fitDistortion(mp, k, reference_frame_number=refFrameNumber, reference_point=referencePoint,
                          use_position_uncertainties=usePositionUncertainties,
                          use_reduced_coordinates=useReducedCoordinates)
    lazAC.display_results(scale_factor_for_residuals=1.)
    lazAC.plotResiduals(evaluation_frame_number, outDir, nameSeed, omc_scale=1., save_plot=saveplot, omc_unit='mas')
    lazAC.plotResults(evaluation_frame_number, outDir, nameSeed, saveplot=saveplot)

    print('Simulation: Rotated by angle {0:1.1f} deg'.format(angle_deg))
    if testSkyPixelMatch:
        print('Simulation: Pixel scale in x {0:1.6f} deg/pix'.format(np.ptp(ra_deg_all) / np.ptp(x)))
        print('Simulation: Pixel scale in x {0:1.6f} deg/pix'.format(np.ptp(de_deg_all) / np.ptp(y)))

    return lazAC


def testDistortionFitLMC():
    print('*' * 30, ' TEST DISTORTION FIT  -- LMC', '*' * 30)
    deg2mas = u.deg.to(u.mas)

    # center of test field in RA and Dec
    RA0_deg = 80.
    DE0_deg = -70.
    cosdec = np.cos(np.deg2rad(DE0_deg))

    # coordinates on grid
    Ngrid = 10
    fieldsize = 2. * u.arcmin
    ra_max_deg = fieldsize.to(u.deg).value / 2. / cosdec
    de_max_deg = fieldsize.to(u.deg).value / 2.

    ra_deg = np.linspace(RA0_deg - ra_max_deg, RA0_deg + ra_max_deg, Ngrid)
    de_deg = np.linspace(DE0_deg - de_max_deg, DE0_deg + de_max_deg, Ngrid)
    ra_deg_mesh, de_deg_mesh = np.meshgrid(ra_deg, de_deg)
    ra_deg_all = ra_deg_mesh.flatten()
    de_deg_all = de_deg_mesh.flatten()

    # project to detector pixel coordinates
    scale = deg2mas
    x, y = RADec2Pix_TAN(ra_deg_all, de_deg_all, RA0_deg, DE0_deg, scale)

    # prepare regular coordinate grid
    T = Table((x, y), names=('x', 'y'))
    T['Id'] = range(len(T))

    # rotation angle between the two catalogs
    angle_deg = -4.
    #     angle_deg = 0.

    # origin of rotation
    origin = (0., 0.)

    # prepare rotated coordinate grid table
    x2, y2 = rotate(origin, T['x'], T['y'], np.deg2rad(angle_deg))
    T2 = Table((x2, y2), names=('x', 'y'))
    T2['Id'] = range(len(T2))

    Nstars = len(T)

    #     introduce measurement noise
    if 1 == 1:
        astrometric_uncertainty = 1.0 * u.milliarcsecond
        np.random.seed(0)
        T['x'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value * scale, Nstars)
        np.random.seed(1)
        T['y'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value * scale, Nstars)
        np.random.seed(2)
        T2['x'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value * scale, Nstars)
        np.random.seed(3)
        T2['y'] += np.random.normal(0., astrometric_uncertainty.to(u.deg).value * scale, Nstars)

    # display vector plot
    X, Y = T['x'], T['y']
    U, V = T2['x'] - T['x'], T2['y'] - T['y']
    pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
    pl.clf()
    Q = pl.quiver(X, Y, U, V, angles='xy')
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.axis('equal')
    pl.title('Difference between catalog positions')
    pl.show()

    colNames = np.array(['x', 'y', 'id', 'original_id', 'sigma_x', 'sigma_y'])
    p = np.zeros((2, Nstars, len(colNames)))
    xmatchId = np.array(T['Id'])
    pos_uncertainty0 = astrometric_uncertainty.to(u.deg).value * scale
    pos_uncertainty1 = astrometric_uncertainty.to(u.deg).value * scale

    x1 = np.array(T['x'])
    x2 = np.array(T2['x'])
    y1 = np.array(T['y'])
    y2 = np.array(T2['y'])
    id1 = np.array(T['Id'])
    id2 = np.array(T2['Id'])
    for i in range(Nstars):
        p[0, i, :] = [x1[i], y1[i], xmatchId[i], id1[i], pos_uncertainty0, pos_uncertainty0]  # cat 1
        p[1, i, :] = [x2[i], y2[i], xmatchId[i], id2[i], pos_uncertainty1, pos_uncertainty1]  # cat 2

    referencePoint = None

    ###############################
    k = 6
    refFrameNumber = 0
    evaluation_frame_number = 1  # using linearFit only takes into account positional uncertainties in evalFrame
    #     verbose = 1
    verboseFigures = 0
    saveplot = 0
    usePositionUncertainties = 1
    useReducedCoordinates = 0  # use coordinate differences relative to reference frame in the fit

    # define the reference point for the differential coordinates. has to have same units as x and y columns, if set to (0,0) the differential coordinates are the same as the coordinates
    # either targetId or reference_point have to be set in the call to fitDistortion
    #     reference_point = [0.,0.]
    referencePoint = np.array([[0., 0.], [0., 0.]])
    ###############################

    nameSeed = 'test'
    outDir = os.path.join(os.path.dirname(__file__), nameSeed)

    # call object constructor
    mp = multiEpochAstrometry(p, colNames)

    considerUncertaintiesInXandY = 0
    lazAC = fitDistortion(mp, k, reference_frame_number=refFrameNumber, evaluation_frame_number=evaluation_frame_number,
                          reference_point=referencePoint, use_position_uncertainties=usePositionUncertainties,
                          use_reduced_coordinates=useReducedCoordinates,
                          consider_uncertainties_in_XY=considerUncertaintiesInXandY)
    lazAC.display_results(evaluation_frame_number=evaluation_frame_number, scale_factor_for_residuals=1., display_correlations=0)
    considerUncertaintiesInXandY = 1
    lazAC_2 = fitDistortion(mp, k, reference_frame_number=refFrameNumber, evaluation_frame_number=evaluation_frame_number,
                            reference_point=referencePoint, use_position_uncertainties=usePositionUncertainties,
                            use_reduced_coordinates=useReducedCoordinates,
                            consider_uncertainties_in_XY=considerUncertaintiesInXandY)
    lazAC_2.display_results(evaluation_frame_number=evaluation_frame_number, scale_factor_for_residuals=1., display_correlations=0)

    xy_unit = u.arcmin
    xy_scale = u.milliarcsecond.to(xy_unit)
    xy_unitStr = xy_unit.to_string()
    lazAC.plotResiduals(evaluation_frame_number, outDir, nameSeed, omc_scale=1., save_plot=saveplot, omc_unit='mas',
                        xy_scale=xy_scale, xy_unit=xy_unitStr)
    lazAC.plotResults(evaluation_frame_number, outDir, nameSeed, saveplot=saveplot, xy_scale=xy_scale, xy_unit=xy_unitStr)

    print('Simulation: Rotated by angle {0:1.1f} deg'.format(angle_deg))
    return lazAC


def construct_polynomial(Cs, P, partial_mode=0, includeAllHigherOrders=True):
    if partial_mode == 0:
        # full model
        polynomial = Matrix(Cs.T) * Matrix(P)
    else:
        minMode = partial_mode - 2
        maxMode = partial_mode
        startIndex = np.int(minMode * (minMode + 2) / 8)
        stopIndex = np.int(maxMode * (maxMode + 2) / 8)
        print('minMode = %d, Startindex = %d, maxMode = %d, stopindex = %d' % (minMode, startIndex, maxMode, stopIndex))
        if includeAllHigherOrders:
            polynomial = Matrix(Cs[startIndex:].T) * Matrix(P[startIndex:])
        else:
            polynomial = Matrix(Cs[startIndex:stopIndex].T) * Matrix(P[startIndex:stopIndex])

    return polynomial


def construct_partial_derivatives_siaf(polynomial, degree, coefficients_x, coefficients_y):
    #     ii = evaluation_frame_number
    n_parameter = np.int((degree + 1) * (degree + 2) / 2)
    #     replacements_X = (','.join(['("p%d",self.Alm[%d,%d])'%(d,ii,d) for d in range(self.Nalm)]))
    #     replacements_Y = (','.join(['("p%d",self.Alm[%d,self.Nalm+%d])'%(d,ii,d) for d in range(self.Nalm)]))

    replacements_X = (','.join(['("p%d",coefficients_x[%d])' % (d, d) for d in range(n_parameter)]))
    replacements_Y = (','.join(['("p%d",coefficients_y[%d])' % (d, d) for d in range(n_parameter)]))

    dfexpr_b = sympy.diff(polynomial, x)
    dfexpr_c = sympy.diff(polynomial, y)
    dfexpr_e = sympy.diff(polynomial, x)
    dfexpr_f = sympy.diff(polynomial, y)

    dffunc_b = dfexpr_b.subs(eval('[' + replacements_X + ']'))
    dffunc_c = dfexpr_c.subs(eval('[' + replacements_X + ']'))
    dffunc_e = dfexpr_e.subs(eval('[' + replacements_Y + ']'))
    dffunc_f = dfexpr_f.subs(eval('[' + replacements_Y + ']'))

    b_func = lambdify((x, y), dffunc_b, 'numpy')
    c_func = lambdify((x, y), dffunc_c, 'numpy')
    e_func = lambdify((x, y), dffunc_e, 'numpy')
    f_func = lambdify((x, y), dffunc_f, 'numpy')

    return b_func, c_func, e_func, f_func


def display_RotScaleSkew(coefficients_x, coefficients_y, verbose=False):
    # number of free parameters
    n_parameter = len(coefficients_x)

    # polynomial degree
    degree = np.int((np.sqrt(8 * n_parameter + 1) - 3) / 2)

    k = 2 * (degree + 1)

    P = sympy.symbols('p0:%d' % n_parameter)
    Cs, polynomialTermOrder = bivariate_polynomial(x, y, k)
    polynomial = construct_polynomial(Cs, P)
    if verbose:
        print('Polynomial term order: {}'.format(polynomialTermOrder))

    b_func, c_func, e_func, f_func = construct_partial_derivatives_siaf(polynomial, degree, coefficients_x,
                                                                        coefficients_y)

    # evaluation point
    x_mesh = 0.
    y_mesh = 0.

    b = b_func(x_mesh, y_mesh)[0][0]
    c = c_func(x_mesh, y_mesh)[0][0]
    e = e_func(x_mesh, y_mesh)[0][0]
    f = f_func(x_mesh, y_mesh)[0][0]

    xoffset = coefficients_x[0]
    yoffset = coefficients_y[0]
    xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis, skew_offaxis, rotation2 = computeRotScaleSkewFromPolyCoeff(
        b, c, e, f)
    params = [xoffset, yoffset, xmag, ymag, xrotation, yrotation, rotation, skew, relative_scale, skew_onaxis,
              skew_offaxis, rotation2]
    param_names = ['xoffset', 'yoffset', 'xmag', 'ymag', 'xrotation', 'yrotation', 'rotation', 'skew', 'relative_scale',
                   'skew_onaxis', 'skew_offaxis', 'rotation2']

    for j in range(len(params)):
        print('%s \t %3.4f' % (param_names[j], params[j]))




