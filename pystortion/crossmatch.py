"""Functions to facilitate crossmatching of sky catalogs.

Authors
-------
    Johannes Sahlmann


References
----------
    This module is build on top of the ``astropy.coordinates`` package and its `search_around_sky`
    method.

"""
import copy
import pickle
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pylab as pl
from pysiaf.utils import projection

from .distortion import multiEpochAstrometry, fitDistortion

def crossmatch_sky_catalogues_with_iterative_distortion_correction(input_source_catalog,
                                                                   input_reference_catalog,
                                                                   out_dir,
                                                                   reference_uncertainty_catalog=None,
                                                                   reference_point_for_projection=None,
                                                                   name_seed='iterative_xmatch',
                                                                   overwrite=True,
                                                                   verbose=False,
                                                                   verbose_figures = False,
                                                                   max_iterations=10,
                                                                   n_interation_switch=5,
                                                                   initial_crossmatch_radius=0.3 * u.arcsec,
                                                                   save_plot=True,
                                                                   adaptive_xmatch_radius_factor=10,
                                                                   k=4,
                                                                   rejection_level_sigma=5.,
                                                                   retain_best_of_multiple_matches=True):
    """

    :param verbose_figures:
    :param input_source_catalog:
    :param input_reference_catalog:
    :param uncertainty_catalog:
    :param reference_point_for_projection:
    :param name_seed:
    :param overwrite:
    :param verbose:
    :param max_iterations:
    :param n_interation_switch:
    :param initial_crossmatch_radius:
    :param k:
    :return:
    """

    if reference_point_for_projection is None:
        reference_point_for_projection = SkyCoord(ra=np.mean(input_reference_catalog.ra), dec = np.mean(input_reference_catalog.dec))

    # distortion fit parameters
    reference_frame_number = 0
    evaluation_frame_number = 1
    use_position_uncertainties = 0

    # scale for tangent plane projection
    # we want to work in milliarcseconds, scale = deg2mas
    scale = reference_point_for_projection.ra.unit.to(u.milliarcsecond)

    # scale_factor_for_residuals
    scale_factor_for_residuals = 1.

    source_catalog_table = Table([input_source_catalog.ra, input_source_catalog.dec], names=('ra', 'dec'))
    reference_catalog_table = Table([input_reference_catalog.ra, input_reference_catalog.dec], names=('ra', 'dec'))

    reference_catalog = copy.deepcopy(input_reference_catalog)

    # this is the source catalog that is not altered
    source_catalog = SkyCoord(ra=source_catalog_table['ra'].data.data * u.degree, dec=source_catalog_table['dec'].data.data * u.degree)

    # project both catalogs onto tangent plane with reference point in the field center, distortion is computed between those projections
    if verbose:
            print('Reference point position RA/Dec {0:3.8f} / {1:3.8f}'.format(reference_point_for_projection.ra,
                                                                       reference_point_for_projection.dec))
    # tangent plane projection
    reference_catalog_x, reference_catalog_y = projection.project_to_tangent_plane(reference_catalog.ra.to(u.deg),
                                                             reference_catalog.dec.to(u.deg),
                                                             reference_point_for_projection.ra,
                                                             reference_point_for_projection.dec, scale)

    #####################################
    # loop to iteratively apply the distortion solution and re-crossmatch the catalog (original x,y coordinates should remain untouched though!)
    # loop that refines the x and y position based on the current iteration's distortion model
    iteration_verbose = True
    iteration_verbose_figures = True
    iteration_verbose = False
    iteration_verbose_figures = False
    iteration_saveplot = save_plot

    xmatch_radius_reducer = True
    xmatch_radius_reducer = False

    iteration_number = 0
    for j in np.arange(max_iterations):
        if iteration_verbose:
            print('-' * 30, ' Iteration %d ' % iteration_number, '-' * 30)

        if iteration_number == 0:
            xmatch_radius = initial_crossmatch_radius
        elif iteration_number > n_interation_switch:
            if xmatch_radius_reducer:
                xmatch_radius = xmatch_radius/2.
            else:
                # adaptive xmatch radius as function of fit residuals
                xmatch_radius = adaptive_xmatch_radius_factor * np.mean(lazAC.rms[1, :] * scale_factor_for_residuals) / 1000. * u.arcsec
                if xmatch_radius > initial_crossmatch_radius:
                    xmatch_radius = initial_crossmatch_radius
        else:
            xmatch_radius = initial_crossmatch_radius
        # if iteration_verbose:
        if verbose:
            print('Using xmatch radius of {}'.format(xmatch_radius))

        # if iteration_number <= n_interation_switch:
        #     # use only bright stars in both image and catalog
        #     T = T_bright.copy()
        #     #             bright_calibration_star_index = np.where(calibFieldCat.table['vcal'] < bright_star_limit_vcal)[0]
        #     calibFieldCat.selectSources('vcal', '<', bright_star_limit_vcal)
        #     #             calibCat   = SkyCoord(ra=calibFieldCat.table['ra'][bright_calibration_star_index]*u.degree, dec=calibFieldCat.table['dec'][bright_calibration_star_index]*u.degree)
        # else:
        #     # use all available stars
        #     T = T_complete.copy()
        #     #             calibFieldCat = calibFieldCat_complete
        #     calibFieldCat.table = calibFieldCat_complete_table


        if iteration_verbose:
            print('using %d sources for distortion fitting (%d stars in reference catalog)' % (
                len(source_catalog_table), len(reference_catalog_table)))



        if iteration_number == 0:
            # primaryCat ra,dec are only used for xmatch
            source_catalog_table['ra_corr']  = source_catalog_table['ra']
            source_catalog_table['dec_corr'] = source_catalog_table['dec']

            # if iteration_number != (max_iterations - 1):
            tmp_verbose = iteration_verbose
            tmp_verbose_figures = iteration_verbose_figures
            tmp_save_plot = iteration_saveplot

        if iteration_number == (max_iterations - 1):
            tmp_verbose = verbose
            tmp_verbose_figures = verbose_figures
            tmp_save_plot = save_plot

        if iteration_number > 0:
            # apply the distortion model to the initial x-y coordinates to improve the xmatch, increase the number of sources

            if verbose:
                print('Polynomial fit residuals of previous iteration: %3.3e native = %3.3f mas (offsets %3.3f / %3.3f)' % (np.mean(lazAC.rms[1, :]), np.mean(lazAC.rms[1, :]) * scale_factor_for_residuals, lazAC.Alm[evaluation_frame_number][0], lazAC.Alm[evaluation_frame_number][lazAC.Nalm]))

            if 0:
                # pl.close('all')
                # k = lazAC.k
                # referencePositionX = referencePoint[:, 0]
                # referencePositionY = referencePoint[:, 1]
                # x_dif = np.array(T['x']) - referencePositionX[refFrameNumber]
                # y_dif = np.array(T['y']) - referencePositionY[refFrameNumber]

                # these parameters are always computed in the reference frame
                ximinusx0 = source_catalog_x - reference_point[reference_frame_number, 0]
                yiminusy0 = source_catalog_y - reference_point[reference_frame_number, 1]
                # compute polynomial terms for all detected sources
                C, polynomialTermOrder = bivariate_polynomial(ximinusx0, yiminusy0, k)

                # since we are not using reduced coordinates, the model position is simply
                PHIx = np.array(C.T * np.mat(lazAC.Alm[evaluation_frame_number, 0:lazAC.Nalm ]).T).flatten();
                PHIy = np.array(C.T * np.mat(lazAC.Alm[evaluation_frame_number,   lazAC.Nalm:]).T).flatten();

                #  idl_tan_x  (x_corr and y_corr)      are in IDL coordinates (degrees, tangent-plane projected IDL frame)
                # distortion corrected tangent-plane coordinates
                source_catalog_x_corr = PHIx
                source_catalog_y_corr = PHIy
            else:
                source_catalog_x_corr, source_catalog_y_corr = lazAC.apply_polynomial_transformation(evaluation_frame_number, source_catalog_x, source_catalog_y)
                # 1/0
                # pl.figure()
                # pl.plot(source_catalog_x_corr, source_catalog_y_corr, 'bo')
                # pl.plot(source_catalog_x, source_catalog_y, 'ko', mfc=None)
                # pl.show()

            #  deproject    to RA/Dec (now distortion corrected)
            source_catalog_RA_corr, source_catalog_Dec_corr = projection.deproject_from_tangent_plane(source_catalog_x_corr, source_catalog_y_corr, reference_point_for_projection.ra, reference_point_for_projection.dec, scale)

            source_catalog_table['ra_corr'] = source_catalog_RA_corr
            source_catalog_table['dec_corr'] = source_catalog_Dec_corr

            if 0:
                pl.figure()
                pl.plot(source_catalog_table['ra_corr'], source_catalog_table['dec_corr'], 'bo', label='corrected')
                pl.plot(source_catalog_table['ra'], source_catalog_table['dec'], 'ko', mfc='w', label='original')
                pl.plot(reference_catalog_table['ra'], reference_catalog_table['dec'], 'r.', mfc='w', label='reference')
                pl.legend(loc='best')
                pl.show()
                1/0

        #####################################
        # do the crossmatch
        source_catalog_for_crossmatch = SkyCoord(ra=source_catalog_table['ra_corr'].data.data * u.degree, dec=source_catalog_table['dec_corr'].data.data * u.degree)
        iteration_name_seed = '%s_iteration%d' % (name_seed, iteration_number)

        # run xmatch
        pickle_file = os.path.join(out_dir, 'xmatch_%s.pkl' % iteration_name_seed)
        if (not os.path.isfile(pickle_file)) | (overwrite):
            index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de = xmatch(source_catalog_for_crossmatch,
                                                                                           reference_catalog,
                                                                                           xmatch_radius,
                                                                                           rejection_level_sigma,
                                                                                           retain_best_match=retain_best_of_multiple_matches,
                                                                                           verbose=tmp_verbose,
                                                                                           verbose_figures=tmp_verbose_figures,
                                                                                           saveplot=tmp_save_plot,
                                                                                           out_dir=out_dir,
                                                                                           name_seed=iteration_name_seed)
            pickle.dump((index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de), open(pickle_file, "wb"))
        else:
            index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de = pickle.load(open(pickle_file, "rb"))
            if verbose:
                print("Loaded pickled file  %s" % pickle_file);

        # if iteration_verbose:
        if verbose:
            print('{} cross-matched sources (mean xmatch distance in RA / Dec {:3.3f} / {:3.3f})'.format(len(index_source_cat), np.mean(diff_raStar).to(u.milliarcsecond), np.mean(diff_de).to(u.milliarcsecond)))
        if max_iterations == 1:
            break



        ############################################################
        # PREPARE DISTORTION FIT
        n_stars = len(index_source_cat)
        col_names = np.array(['x', 'y', 'sigma_x', 'sigma_y', 'id'])
        p = np.zeros((2, n_stars, len(col_names)))
        mp = multiEpochAstrometry(p, col_names)

        i_x = np.where(mp.colNames == 'x')[0][0]
        i_y = np.where(mp.colNames == 'y')[0][0]


        # tangent plane projection
        source_catalog_x, source_catalog_y = projection.project_to_tangent_plane(source_catalog.ra.to(u.deg), source_catalog.dec.to(u.deg),
                                                                 reference_point_for_projection.ra,
                                                                 reference_point_for_projection.dec, scale)

        # astropy3 support, if RADec2Pix_TAN returns astropy quantities, strip those
        try:
            unit = source_catalog_x.unit
            source_catalog_x = source_catalog_x.value
            source_catalog_y = source_catalog_y.value
        except AttributeError:
            pass

        # first catalog, these are the measured sources because we want to determine tha transformation that corrects onto the reference catalog frame
        mp.p[0, :, [i_x, i_y]] = np.vstack((source_catalog_x[index_source_cat], source_catalog_y[index_source_cat]))
        # second catalog (here this is the reference catalog)
        mp.p[1, :, [i_x, i_y]] = np.vstack((reference_catalog_x[index_reference_cat], reference_catalog_y[index_reference_cat]))

        ############################################################
        # DISTORTION FIT

        # define the reference point for the differential coordinates. has to have same units as x and y columns, if set to (0,0) the differential coordinates are the same as the coordinates
        # either targetId or referencePoint have to be set in the call to fitDistortion
        reference_point = np.array([[0., 0.], [0., 0.]])

        lazAC = fitDistortion(mp, k, reference_frame_number=reference_frame_number, evaluation_frame_number=evaluation_frame_number,
                                           reference_point=reference_point,
                                           use_position_uncertainties=use_position_uncertainties)

        if (iteration_number == (max_iterations - 1)) & (verbose):
        # if (verbose):
            lazAC.display_results(evaluation_frame_number=evaluation_frame_number, scale_factor_for_residuals=1.)
            lazAC.plotResiduals(evaluation_frame_number, out_dir, name_seed, omc_scale = 1., save_plot=save_plot, omc_unit='mas')

        iteration_number += 1

    return index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de



def xmatch(primary_cat, secondary_cat, xmatch_radius, rejection_level_sigma=0,
           remove_multiple_matches=True, retain_best_match=False, verbose=False,
           verbose_figures=False, saveplot=False, out_dir=None, name_seed=None):
    """Crossmatch two SkyCoord catalogs with RA and Dec fields.

    Parameters
    ----------
    primary_cat : SkyCoord catalog
        primary catalog, the sources in the secondary catalog will be searched for closest match
        to sources in primary catalog
    secondary_cat : SkyCoord catalog
        secondary catalog
    xmatch_radius : float with astropy angular unit
        angular radius for the crossmatch
    rejection_level_sigma : float
        outlier rejection level in terms of dispersion of the crossmatch distance
    remove_multiple_matches : bool
        Whether to remove any entry that has multiple matches
    retain_best_match : bool
        Whether to retain the closest match for entries with multiple matches
    verbose : bool
        verbosity
    verbose_figures : bool
        Whether to display supporting figures
    saveplot : bool
        Whether to save figures to disk
    out_dir : str
        Target directory for saving figures
    name_seed : str
        naming seed for figures

    Returns
    -------
    index_primary_cat, index_secondary_cat, d2d, d3d, diff_raStar, diff_de : tuple
        Results of crossmatch

    History
    -------
    First version written 2016-12-22 by J. Sahlmann, AURA/STScI

    """
    # find sources in secondary_cat that are closest to sources in primary_cat
    index_secondary_cat, index_primary_cat, d2d, d3d = primary_cat.search_around_sky(secondary_cat,
                                                                                     xmatch_radius)

    if len(index_secondary_cat) == 0:
        raise RuntimeError('Crossmatch failed, no matches found.')

    if verbose:
        print('xmatch: found {} matches'.format(len(index_secondary_cat)))

    # reject outliers
    if rejection_level_sigma != 0:
        index_primary_cat, index_secondary_cat, d2d, d3d = \
            remove_xmatch_outliers(primary_cat, secondary_cat, index_primary_cat,
                                   index_secondary_cat, d2d, d3d, rejection_level_sigma,
                                   verbose=verbose)
        if verbose:
            print('xmatch: {} matches after outlier rejection'.format(len(index_secondary_cat)))

    # clean up multiple matches
    if remove_multiple_matches:
        index_secondary_cat, index_primary_cat, d2d, d3d = \
            remove_xmatch_multiples(index_secondary_cat, index_primary_cat, d2d, d3d,
                                    retain_best_match=retain_best_match, verbose=verbose)

        if verbose:
            print('xmatch: {} matches (after first multiple match rejection)'.format(
                len(index_secondary_cat)))

        index_primary_cat, index_secondary_cat, d2d, d3d = \
            remove_xmatch_multiples(index_primary_cat, index_secondary_cat, d2d, d3d,
                                    retain_best_match=retain_best_match, verbose=verbose)
        if verbose:
            print('xmatch: {} matches (after second multiple match rejection)'.format(
                len(index_secondary_cat)))

    if len(index_primary_cat) == 0:
        raise RuntimeError('xmatch: crossmatch did not return any match')

    if verbose_figures:
        fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
        pl.clf()
        if len(secondary_cat.ra) >= len(primary_cat.ra):
            primary_catalog_plot_symbol = 'bo'
            # primary_catalog_plot_mfc = None
            secondary_catalog_plot_symbol = 'r.'

            # primary_ms = 3
            primary_zorder = -50
            secondary_zorder = -40

        else:
            primary_catalog_plot_symbol = 'bo'
            secondary_catalog_plot_symbol = 'r.'
            # primary_ms = 0.5
            primary_zorder = -50
            secondary_zorder = -40

        pl.plot(secondary_cat.ra, secondary_cat.dec, secondary_catalog_plot_symbol,
                label='secondary catalog', zorder=secondary_zorder, mfc=None)
        pl.plot(primary_cat.ra, primary_cat.dec, primary_catalog_plot_symbol,
                label='primary catalog', zorder=primary_zorder, mfc='none', ms=10, mew=2)
        pl.plot(secondary_cat.ra[index_secondary_cat], secondary_cat.dec[index_secondary_cat], 'kx',
                label='xmatch sources', zorder=-20)
        ax = pl.gca()
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        pl.xlabel('RA (deg)')
        pl.ylabel('Dec (deg)')
        pl.legend()
        pl.show()
        if saveplot:
            figName = os.path.join(out_dir, '%s_xmatch_onSky.pdf' % name_seed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    # define quantities for quality check and further processing
    cosDecFactor = np.cos(np.deg2rad(primary_cat.dec[index_primary_cat]))
    diff_raStar = (secondary_cat.ra[index_secondary_cat] - primary_cat.ra[index_primary_cat])\
                  * cosDecFactor
    diff_de = (secondary_cat.dec[index_secondary_cat] - primary_cat.dec[index_primary_cat])

    xmatchDistance = d2d.copy().to(u.milliarcsecond)

    # display actual distortion
    if verbose_figures:
        X = primary_cat.ra[index_primary_cat]
        Y = primary_cat.dec[index_primary_cat]
        #         zeroPointIndex = np.where(diff_raStar == np.median(diff_raStar))[0][0]
        #         zeroPointIndex = np.where(diff_de == np.median(diff_de))[0][0]
        U0 = diff_raStar
        V0 = diff_de
        U = U0 - np.median(U0)
        V = V0 - np.median(V0)

        UV_factor = primary_cat.ra.unit.to(u.milliarcsecond)

        n_bins = np.int(len(xmatchDistance) / 5)

        # xmatch diagnostics
        pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k'); pl.clf()
        pl.subplot(1, 2, 1)
        pl.hist(xmatchDistance.value, n_bins)
        pl.xlabel('Crossmatch distance (%s)' % (xmatchDistance.unit))
        pl.subplot(1, 2, 2)
        pl.hist(U0.value * UV_factor, n_bins, color='b', label='X')
        pl.hist(V0.value * UV_factor, n_bins, color='r', alpha=0.5, label='Y')
        pl.xlabel('Coordinate Difference in X and Y {}'.format(u.milliarcsecond))
        pl.legend(loc='best')
        fig.tight_layout(h_pad=0.0)
        pl.show()
        if saveplot:
            figName = os.path.join(out_dir, '%s_xmatch_distance.pdf' % name_seed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

        if 0 == 1:
            zeroPointIndex = np.where(np.abs(diff_de) == np.median(np.abs(diff_de)))[0][0]
            U0 = diff_raStar;
            V0 = diff_de;
            U = U0 - U0[zeroPointIndex]
            V = V0 - V0[zeroPointIndex]
            print('Zeropoint U0 {0:3.1f}'.format(U0[zeroPointIndex].to(u.milliarcsecond)))
            print('Zeropoint V0 {0:3.1f}'.format(V0[zeroPointIndex].to(u.milliarcsecond)))
            print('U0 : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(np.min(U0).to(u.milliarcsecond),
                                                                          np.max(U0).to(u.milliarcsecond),
                                                                          np.median(U0).to(u.milliarcsecond)))
            print('V0 : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(np.min(V0).to(u.milliarcsecond),
                                                                          np.max(V0).to(u.milliarcsecond),
                                                                          np.median(V0).to(u.milliarcsecond)))
            print('U  : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(np.min(U).to(u.milliarcsecond),
                                                                          np.max(U).to(u.milliarcsecond),
                                                                          np.median(U).to(u.milliarcsecond)))
            print('V  : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(np.min(V).to(u.milliarcsecond),
                                                                          np.max(V).to(u.milliarcsecond),
                                                                          np.median(V).to(u.milliarcsecond)))

            naiveDistanceModulus = np.sqrt(U0 ** 2 + V0 ** 2)
            DistanceModulus = np.abs(primary_cat[index_primary_cat].separation(secondary_cat[index_secondary_cat]))

            print('naiveDistanceModulus  : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(
                np.min(naiveDistanceModulus).to(u.milliarcsecond), np.max(naiveDistanceModulus).to(u.milliarcsecond),
                np.median(naiveDistanceModulus).to(u.milliarcsecond)))
            print('     DistanceModulus  : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(
                np.min(DistanceModulus).to(u.milliarcsecond), np.max(DistanceModulus).to(u.milliarcsecond),
                np.median(DistanceModulus).to(u.milliarcsecond)))


            pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
            pl.clf()
            pl.hist(U.to(u.milliarcsecond), 100, color='r')
            pl.hist(V.to(u.milliarcsecond), 100, color='b')
            pl.show()

            pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
            pl.clf()
            pl.plot(DistanceModulus.to(u.milliarcsecond), naiveDistanceModulus.to(u.milliarcsecond), 'b.')
            pl.show()

        fig = pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k')
        pl.clf()

        pl.subplot(1, 2, 1)
        #         headlength = 10
        forGaia = 1
        if forGaia:
            scale1 = 0.003
            scale2 = 0.0005

        if forGaia:
            Q = pl.quiver(X, Y, U0, V0, angles='xy', scale_units='xy', scale=scale1)
        else:
            Q = pl.quiver(X, Y, U0, V0, angles='xy', scale_units='xy')

        # Q = pl.quiver(X[::3],Y[::3], U0[::3], V0[::3], angles='xy',units='inches')


        gmc_ra = np.mean(pl.xlim())
        gmc_de = np.mean(pl.ylim())
        #         cosdec = np.cos(np.deg2rad(gmc_de))
        size_deg = pl.xlim()[1] - gmc_ra - 0.1
        #         ra_min = gmc_ra - (size_deg / cosdec)
        #         ra_max = gmc_ra + (size_deg / cosdec)
        de_min = gmc_de - size_deg
        de_max = gmc_de + size_deg

        pl.ylim((de_min, de_max))
        #         1/0
        ax = pl.gca()  # ;
        # pl.axis('equal')
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        pl.xlabel('Right Ascension (deg)');
        pl.ylabel('Declination (deg)');
        pl.title('Difference between catalog positions')

        pl.subplot(1, 2, 2)
        if forGaia:
            Q = pl.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale2)
        else:
            Q = pl.quiver(X, Y, U, V, angles='xy', scale_units='xy')

        # Q = pl.quiver(X[::3],Y[::3], U[::3], V[::3], angles='xy',units='inches')
        pl.ylim((de_min, de_max))
        ax = pl.gca()  # ;
        # pl.axis('equal')
        ax.invert_xaxis()
        pl.xlabel('Right Ascension (deg)');
        pl.ylabel('Declination (deg)');
        pl.title('Average offset subtracted')
        fig.tight_layout(h_pad=0.0)
        pl.show()

        if saveplot:
            figName = os.path.join(out_dir, '%s_xmatch_distortionActual.pdf' % name_seed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

    return index_primary_cat, index_secondary_cat, d2d, d3d, diff_raStar, diff_de


def remove_xmatch_multiples(index_primary, index_secondary, d2d, d3d, retain_best_match=True,
                            verbose=False):
    """Return crossmatches where multiple matches have been cleaned up.

    Parameters
    ----------
    index_primary
    index_secondary
    d2d
    d3d
    retain_best_match
    verbose

    Returns
    -------
    index_primary, index_secondary, d2d, d3d : tuple

    """
    n_entries = len(index_primary)
    n_unique_entries = len(np.unique(index_primary))
    if verbose:
        print('xmatch: cleanMultipleCrossMatches: there are {0:d} unique entries in the primary '
              'catalog'.format(n_unique_entries))

    if n_entries != n_unique_entries:
        # get array of unique indices (here like identifier) and the corresponding occurrence count
        unique_array, return_counts = np.unique(index_primary, return_counts=True)

        # indices in unique array of stars with multiple crossmatches
        index_in_unique_array = np.where(return_counts > 1)[0]
        number_of_stars_with_multiplematches = len(index_in_unique_array)
        if verbose:
            if retain_best_match:
                print('xmatch: result contains {} multiple matches affecting {} entries, '
                      'keeping closest match'.format(number_of_stars_with_multiplematches,
                                                     n_entries - n_unique_entries))
            else:
                print('xmatch: result contains {} multiple matches affecting {} entries, removing '
                      'them all'.format(number_of_stars_with_multiplematches,
                                        n_entries - n_unique_entries))

        # identifiers (i.e. numbers in index_primary) of stars with multiple crossmatches
        multi_matches = unique_array[index_in_unique_array]

        # index in index_primary where multiple matches occur
        index_multiple_match = np.where(np.in1d(index_primary, multi_matches))[0]
        if verbose:
            print(d2d[index_multiple_match], index_primary[index_multiple_match], index_secondary[index_multiple_match])

        good_index_in_index_primary = np.zeros(number_of_stars_with_multiplematches)
        for ii, jj in enumerate(multi_matches):
            tmp_idx0 = np.where(index_primary[index_multiple_match] == jj)[0]
            tmp_idx1 = np.argmin(d2d[index_multiple_match][tmp_idx0])
            good_index_in_index_primary[ii] = index_primary[index_multiple_match][tmp_idx0[tmp_idx1]]

        if retain_best_match:
            index_to_remove = np.setdiff1d(index_multiple_match, good_index_in_index_primary)
        else:
            index_to_remove = index_multiple_match

        mask = np.ones(len(index_primary), dtype=bool)
        mask[index_to_remove] = False
        index_primary = index_primary[mask]
        index_secondary = index_secondary[mask]
        d2d = d2d[mask]
        d3d = d3d[mask]
        if len(index_primary) != len(np.unique(index_primary)):
            print('xmatch: Multiple match cleanup procedure failed')
            # get array of unique indices (here like identifier) and the corresponding occurrence count
            unique_array, return_counts = np.unique(index_primary, return_counts=True)

            # indices in unique array of stars with multiple crossmatches
            index_in_unique_array = np.where(return_counts > 1)[0]
            number_of_stars_with_multiplematches = len(index_in_unique_array)
            print('xmatch: result still contains {} multiple matches'.format(
                number_of_stars_with_multiplematches))

    return index_primary, index_secondary, d2d, d3d


def remove_xmatch_outliers(primary_cat, secondary_cat, index_primary, index_secondary, d2d, d3d,
                           rejection_level_sigma, verbose=False):
    """Remove crossmatched sources that have separations beyond rejection_level_sigma.

    Parameters
    ----------
    primary_cat
    secondary_cat
    index_primary
    index_secondary
    d2d
    d3d
    rejection_level_sigma
    verbose

    Returns
    -------
    index_primary, index_secondary, d2d, d3d : tuple

    """
    if len(index_primary) == 0:
        raise RuntimeError('Invalid crossmatch.')

    # first step: consider only separation
    d2d_median = np.median(d2d)
    d2d_std = np.std(d2d)
    idx_outlier = np.where(np.abs(d2d - d2d_median) > rejection_level_sigma * d2d_std)[0]
    mask = np.ones(len(index_primary), dtype=bool)
    mask[idx_outlier] = False
    index_primary = index_primary[mask]
    index_secondary = index_secondary[mask]
    d2d = d2d[mask]
    d3d = d3d[mask]
    if verbose:
        print('xmatch: Removed {} entries as crossmatch outlier beyond {:3.1f} sigma (separation only)'.format(
            len(idx_outlier), rejection_level_sigma))

    # second step: consider separation in RA* and Dec
    delta_alpha_star = (secondary_cat.ra[index_secondary] - primary_cat.ra[index_primary]) * np.cos(
        np.deg2rad(primary_cat.dec[index_primary]))
    delta_delta = (secondary_cat.dec[index_secondary] - primary_cat.dec[index_primary])

    delta_alpha_star_median = np.median(delta_alpha_star)
    delta_delta_median = np.median(delta_delta)
    delta_alpha_star_std = np.std(delta_alpha_star)
    delta_delta_std = np.std(delta_delta)

    idx_outlier = np.where((np.abs(delta_alpha_star - delta_alpha_star_median) > rejection_level_sigma * delta_alpha_star_std) | (
        np.abs(delta_delta - delta_delta_median) > rejection_level_sigma * delta_delta_std))[0]
    mask = np.ones(len(index_primary), dtype=bool)
    mask[idx_outlier] = False
    index_primary = index_primary[mask]
    index_secondary = index_secondary[mask]
    d2d = d2d[mask]
    d3d = d3d[mask]
    if verbose:
        print('xmatch: Removed {} entries as crossmatch outlier beyond {:3.1f} sigma (considering '
              'RA* and Dec jointly)'.format(len(idx_outlier), rejection_level_sigma))

    return index_primary, index_secondary, d2d, d3d
