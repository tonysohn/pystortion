
import copy
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import pylab as pl
from matplotlib.ticker import FormatStrFormatter


from .projection import Pix2RADec_TAN, RADec2Pix_TAN

import os, sys
home_dir = os.environ['HOME']
sys.path.append(os.path.join(home_dir,'jwst/code/pyDistortion'))
import pyDistortion



def crossmatch_sky_catalogues_with_iterative_distortion_correction(input_source_catalog,
                                                                   input_reference_catalog,
                                                                   out_dir,
                                                                   reference_uncertainty_catalog=None,
                                                                   reference_point_for_projection=None,
                                                                   name_seed='iterative_xmatch',
                                                                   overwrite=False,
                                                                   verbose=True,
                                                                   verbose_figures = True,
                                                                   max_iterations=10,
                                                                   n_interation_switch=5,
                                                                   initial_crossmatch_radius=0.3 * u.arcsec,
                                                                   save_plot=True,
                                                                   adaptive_xmatch_radius_factor=10,
                                                                   k=4):
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

    # crossmatch parameters
    rejection_sigma_level = 5.0
    retainBestOfMultipleMatches = 1

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
    print('Reference point position RA/Dec {0:3.8f} / {1:3.8f}'.format(reference_point_for_projection.ra,
                                                                       reference_point_for_projection.dec))
    # tangent plane projection
    reference_catalog_x, reference_catalog_y = RADec2Pix_TAN(reference_catalog.ra.to(u.deg),
                                                             reference_catalog.dec.to(u.deg),
                                                             reference_point_for_projection.ra,
                                                             reference_point_for_projection.dec, scale);

    #####################################
    # loop to iteratively apply the distortion solution and re-crossmatch the catalog (original x,y coordinates should remain untouched though!)
    # loop that refines the x and y position based on the current iteration's distortion model
    iteration_verbose = 1
    iteration_verbose_figures = 0
    iteration_saveplot = save_plot

    iteration_number = 0
    for j in np.arange(max_iterations):
        print('-' * 30, ' Iteration %d ' % iteration_number, '-' * 30)

        if iteration_number == 0:
            xmatch_radius = initial_crossmatch_radius
        elif iteration_number > n_interation_switch:
            # adaptive xmatch radius as function of fit residuals
            xmatch_radius = adaptive_xmatch_radius_factor * np.mean(lazAC.rms[1, :] * scale_factor_for_residuals) / 1000. * u.arcsec
        else:
            xmatch_radius = initial_crossmatch_radius / 1.
        if iteration_verbose:
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

            print('Polynomial fit residuals of previous iteration: %3.3e native = %3.3f mas' % (np.mean(lazAC.rms[1, :]), np.mean(lazAC.rms[1, :] * scale_factor_for_residuals)))

            # pl.close('all')
            # apply the distortion model to the initial x-y coordinates of SEP to improve the xmatch, increase the number of sources
            # k = lazAC.k
            # referencePositionX = referencePoint[:, 0]
            # referencePositionY = referencePoint[:, 1]
            # x_dif = np.array(T['x']) - referencePositionX[refFrameNumber]
            # y_dif = np.array(T['y']) - referencePositionY[refFrameNumber]

            # these parameters are always computed in the reference frame
            ximinusx0 = source_catalog_x - reference_point[reference_frame_number, 0]
            yiminusy0 = source_catalog_y - reference_point[reference_frame_number, 1]
            # compute polynomial terms for all detected sources
            C, polynomialTermOrder = pyDistortion.generalBivariatePolynomial(ximinusx0, yiminusy0, k)

            # since we are not using reduced coordinates, the model position is simply
            PHIx = np.array(C.T * np.mat(lazAC.Alm[evaluation_frame_number, 0:lazAC.Nalm ]).T).flatten();
            PHIy = np.array(C.T * np.mat(lazAC.Alm[evaluation_frame_number,   lazAC.Nalm:]).T).flatten();

            #  idl_tan_x  (x_corr and y_corr)      are in IDL coordinates (degrees, tangent-plane projected IDL frame)
            # distortion corrected tangent-plane coordinates
            source_catalog_x_corr = PHIx
            source_catalog_y_corr = PHIy

            #  deproject    to RA/Dec (now distortion corrected)
            source_catalog_RA_corr, source_catalog_Dec_corr = Pix2RADec_TAN(source_catalog_x_corr, source_catalog_y_corr, reference_point_for_projection.ra, reference_point_for_projection.dec, scale)
            # tmp_v2_deg, tmp_v3_deg = distortion.Pix2RADec_TAN(tmp_v2_tan_deg, tmp_v3_tan_deg,
            #                                                   referencePointForProjection_RADec[0],
            #                                                   referencePointForProjection_RADec[1], scale)
            # #             tmp_v2_deg = tmp_v3_deg - 360.



            # v2v3 -> RA/Dec
            # RA_corr, Dec_corr = siaf_rotations.pointing(attitude_ref, tmp_v2_deg * deg2arcsec, tmp_v3_deg * deg2arcsec)

            #             1/0

            # primaryCatTable['ra'] = RA_corr
            # primaryCatTable['dec'] = Dec_corr
            source_catalog_table['ra_corr'] = source_catalog_RA_corr
            source_catalog_table['dec_corr'] = source_catalog_Dec_corr

        #####################################
        # do the crossmatch
        source_catalog_for_crossmatch = SkyCoord(ra=source_catalog_table['ra_corr'].data.data * u.degree, dec=source_catalog_table['dec_corr'].data.data * u.degree)
        iteration_name_seed = '%s_iteration%d' % (name_seed, iteration_number)

        # run xmatch
        pickle_file = os.path.join(out_dir, 'xmatch_%s.pkl' % iteration_name_seed)
        if (not os.path.isfile(pickle_file)) | (overwrite):
            index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de = pyDistortion.xmatch(source_catalog_for_crossmatch,
                                                                                                        reference_catalog,
                                                                                               xmatch_radius,
                                                                                               rejection_sigma_level,
                                                                                               retainBestOfMultipleMatches=retainBestOfMultipleMatches,
                                                                                               verbose=tmp_verbose,
                                                                                               verboseFigures=tmp_verbose_figures,
                                                                                               saveplot=tmp_save_plot,
                                                                                               outDir=out_dir,
                                                                                               nameSeed=iteration_name_seed)
            pickle.dump((index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de), open(pickle_file, "wb"))
            if verbose:
                print("Wrote pickled file  %s" % pickle_file)
        else:
            index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de = pickle.load(open(pickle_file, "rb"))
            if verbose:
                print("Loaded pickled file  %s" % pickle_file);

        if iteration_verbose:
            print('{} cross-matched sources'.format(len(index_source_cat)))
        if max_iterations == 1:
            break



        ############################################################
        # PREPARE DISTORTION FIT
        n_stars = len(index_source_cat)
        col_names = np.array(['x', 'y', 'sigma_x', 'sigma_y'])
        p = np.zeros((2, n_stars, len(col_names)))
        mp = pyDistortion.multiEpochAstrometry(p, col_names)

        i_x = np.where(mp.colNames == 'x')[0][0]
        i_y = np.where(mp.colNames == 'y')[0][0]


        # tangent plane projection
        source_catalog_x, source_catalog_y = RADec2Pix_TAN(source_catalog.ra.to(u.deg), source_catalog.dec.to(u.deg),
                                                                 reference_point_for_projection.ra,
                                                                 reference_point_for_projection.dec, scale);


        # first catalog (reference)
        mp.p[reference_frame_number, :, [i_x, i_y]] = np.vstack((reference_catalog_x[index_reference_cat], reference_catalog_y[index_reference_cat]))
        # second catalog (source)
        mp.p[evaluation_frame_number, :, [i_x, i_y]] = np.vstack((source_catalog_x[index_source_cat], source_catalog_y[index_source_cat]))

        ############################################################
        # DISTORTION FIT

        # define the reference point for the differential coordinates. has to have same units as x and y columns, if set to (0,0) the differential coordinates are the same as the coordinates
        # either targetId or referencePoint have to be set in the call to fitDistortion
        reference_point = np.array([[0., 0.], [0., 0.]])


        lazAC = pyDistortion.fitDistortion(mp, k, reference_frame_number=reference_frame_number, evaluation_frame_number=evaluation_frame_number,
                                           reference_point=reference_point,
                                           use_position_uncertainties=use_position_uncertainties)

        if (iteration_number == (max_iterations - 1)) | (1==0):
            lazAC.displayResults(evalFrameNumber=evaluation_frame_number, scaleFactorForResiduals=1.)
            lazAC.plotResiduals(evaluation_frame_number, out_dir, name_seed, omc_scale = 1., save_plot=save_plot, omc_unit='mas')

        iteration_number += 1

    return index_source_cat, index_reference_cat, d2d, d3d, diff_raStar, diff_de


def cleanMultipleCrossMatches_search_around_sky(index_primary, index_secondary, d2d, d3d, retainBestOfMultipleMatches=1,
                                                verbose=0):
    Nentries = len(index_primary)
    NuniqueEntries = len(np.unique(index_primary))
    if verbose:
        print('xmatch: cleanMultipleCrossMatches: there are {0:d} unique entries in the primary catalog'.format(
            NuniqueEntries))
    if Nentries != NuniqueEntries:

        # get array of unique indices (here like identifier) and the corresponding occurrence count
        uniqueArray, returnCounts = np.unique(index_primary, return_counts=True)
        # indices in unique array of stars with multiple crossmatches
        idx_inUniqueArray = np.where(returnCounts > 1)[0]
        numberOfStarsWithMultiplematches = len(idx_inUniqueArray)
        if verbose:
            if retainBestOfMultipleMatches:
                print(
                    'xmatch: result contains %d multiple matches affecting %d entries, keeping closest-separation match' % (
                    numberOfStarsWithMultiplematches, Nentries - NuniqueEntries))
            else:
                print('xmatch: result contains %d multiple matches affecting %d entries, removing them all' % (
                numberOfStarsWithMultiplematches, Nentries - NuniqueEntries))

        # identifiers (i.e. numbers in index_primary) of stars with multiple crossmatches
        multiMatches = uniqueArray[idx_inUniqueArray]

        # index in index_primary where multiple matches occur
        idx_multipleMatch = np.where(np.in1d(index_primary, multiMatches))[0]
        if (verbose) & (0 == 1):
            print(d2d[idx_multipleMatch], index_primary[idx_multipleMatch], index_secondary[idx_multipleMatch])

        goodIndexIn_index_primary = np.zeros(numberOfStarsWithMultiplematches)
        for ii, jj in enumerate(multiMatches):
            tmpIdx0 = np.where(index_primary[idx_multipleMatch] == jj)[0]
            tmpIdx1 = np.argmin(d2d[idx_multipleMatch][tmpIdx0])
            goodIndexIn_index_primary[ii] = index_primary[idx_multipleMatch][tmpIdx0[tmpIdx1]]
        # print(ii,jj,tmpIdx1)

        if retainBestOfMultipleMatches:
            idx_toRemove = np.setdiff1d(idx_multipleMatch, goodIndexIn_index_primary)
        else:
            idx_toRemove = idx_multipleMatch

        mask = np.ones(len(index_primary), dtype=bool)
        mask[idx_toRemove] = False
        index_primary = index_primary[mask]
        index_secondary = index_secondary[mask]
        d2d = d2d[mask]
        d3d = d3d[mask]
        if len(index_primary) != len(np.unique(index_primary)):
            print('xmatch: Multiple match cleanup procedure failed')
            # get array of unique indices (here like identifier) and the corresponding occurrence count
            uniqueArray, returnCounts = np.unique(index_primary, return_counts=True)

            # indices in unique array of stars with multiple crossmatches
            idx_inUniqueArray = np.where(returnCounts > 1)[0]
            numberOfStarsWithMultiplematches = len(idx_inUniqueArray)
            print('xmatch: result still contains %d multiple matches' % numberOfStarsWithMultiplematches)

    return index_primary, index_secondary, d2d, d3d


def removeCrossMatchOutliers_search_around_sky(primaryCat, secondaryCat, index_primary, index_secondary, d2d, d3d,
                                               sigmaLevel, verbose=0):
    #   first step: consider only separation
    d2d_median = np.median(d2d)
    d2d_std = np.std(d2d)
    idx_outlier = np.where(np.abs(d2d - d2d_median) > sigmaLevel * d2d_std)[0]
    mask = np.ones(len(index_primary), dtype=bool)
    mask[idx_outlier] = False
    index_primary = index_primary[mask]
    index_secondary = index_secondary[mask]
    d2d = d2d[mask]
    d3d = d3d[mask]
    if verbose:
        print('xmatch: Removed %d entries as crossmatch outlier beyond %3.1f sigma (separation only)' % (
        len(idx_outlier), sigmaLevel))

    # second step: consider separation in RA* and Dec
    diff_raStar = (secondaryCat.ra[index_secondary] - primaryCat.ra[index_primary]) * np.cos(
        np.deg2rad(primaryCat.dec[index_primary]))
    diff_de = (secondaryCat.dec[index_secondary] - primaryCat.dec[index_primary])

    diff_raStar_median = np.median(diff_raStar)
    diff_de_median = np.median(diff_de)
    diff_raStar_std = np.std(diff_raStar)
    diff_de_std = np.std(diff_de)

    idx_outlier = np.where((np.abs(diff_raStar - diff_raStar_median) > sigmaLevel * diff_raStar_std) | (
    np.abs(diff_de - diff_de_median) > sigmaLevel * diff_de_std))[0]
    mask = np.ones(len(index_primary), dtype=bool)
    mask[idx_outlier] = False
    index_primary = index_primary[mask]
    index_secondary = index_secondary[mask]
    d2d = d2d[mask]
    d3d = d3d[mask]
    if verbose:
        print(
            'xmatch: Removed %d entries as crossmatch outlier beyond %3.1f sigma (considering RA* and Dec jointly)' % (
            len(idx_outlier), sigmaLevel))

    return index_primary, index_secondary, d2d, d3d


def xmatch(primaryCat, secondaryCat, xmatchRadius, rejectionSigmaLevel, retainBestOfMultipleMatches=0, verbose=0,
           verboseFigures=1, saveplot=0, outDir=None, nameSeed=None):
    """
    Crossmatch two SkyCoord catalogs with RA and Dec fields
    written 2016-12-22 J. Sahlmann, AURA/STScI

    Parameters
    ----------
    primaryCat : SkyCoord catalog
        primary catalog, the sources in the secondary catalog will be searched for closest match to sources in primary catalog
    secondaryCat : SkyCoord catalog
        secondary catalog
    xmatchRadius : float with astropy angular unit
        angular radius for the crossmatch
    rejectionSigmaLevel : float
        outlier rejection level in terms of dispersion of the crossmatch distance

    """

    # find sources in calibCat that are closest to sources in gaiaCat
    idx_secondaryCat, idx_primaryCat, d2d, d3d = primaryCat.search_around_sky(secondaryCat, xmatchRadius)
    if verbose:
        print('xmatch: %d matches' % len(idx_secondaryCat))
    # idx_secondaryCat, idx_primaryCat  ,  d2d, d3d = removeCrossMatchOutliers_search_around_sky( idx_secondaryCat, idx_primaryCat     ,  d2d, d3d , rejectionSigmaLevel, verbose=verbose)
    #     idx_secondaryCat, idx_primaryCat  ,  d2d, d3d = removeCrossMatchOutliers_search_around_sky( secondaryCat, primaryCat, idx_secondaryCat, idx_primaryCat     ,  d2d, d3d , rejectionSigmaLevel, verbose=verbose)
    if rejectionSigmaLevel != 0:
        idx_primaryCat, idx_secondaryCat, d2d, d3d = removeCrossMatchOutliers_search_around_sky(primaryCat,
                                                                                                secondaryCat,
                                                                                                idx_primaryCat,
                                                                                                idx_secondaryCat, d2d,
                                                                                                d3d,
                                                                                                rejectionSigmaLevel,
                                                                                                verbose=verbose)
        if verbose:
            print('xmatch: %d matches (after outlier rejection)' % len(idx_secondaryCat))

            #     retainBestOfMultipleMatches = 1
    idx_secondaryCat, idx_primaryCat, d2d, d3d = cleanMultipleCrossMatches_search_around_sky(idx_secondaryCat,
                                                                                             idx_primaryCat, d2d, d3d,
                                                                                             retainBestOfMultipleMatches=retainBestOfMultipleMatches,
                                                                                             verbose=verbose)
    if verbose:
        print('xmatch: %d matches (after first multiple rejection rejection)' % len(idx_secondaryCat))
    idx_primaryCat, idx_secondaryCat, d2d, d3d = cleanMultipleCrossMatches_search_around_sky(idx_primaryCat,
                                                                                             idx_secondaryCat, d2d, d3d,
                                                                                             retainBestOfMultipleMatches=retainBestOfMultipleMatches,
                                                                                             verbose=verbose)
    if verbose:
        print('xmatch: %d matches (after second multiple rejection rejection)' % len(idx_secondaryCat))

    # print(len(idx_primaryCat))
    if len(idx_primaryCat) == 0:
        print('WARNING: crossmatch did not return any match')
        1 / 0

    if verboseFigures:
        fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k');
        pl.clf();
        if len(secondaryCat.ra) >= len(primaryCat.ra):
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

        pl.plot(primaryCat.ra, primaryCat.dec, primary_catalog_plot_symbol, label='primary catalog',
                zorder=primary_zorder, mfc=None)  # , ms=primary_ms) #,
        #         pl.plot(primaryCat.ra,primaryCat.dec,'b.',label='primary catalog')
        #         pl.plot(primaryCat.ra[  idx_primaryCat    ],primaryCat.dec[  idx_primaryCat  ],'b.')
        pl.plot(secondaryCat.ra, secondaryCat.dec, secondary_catalog_plot_symbol, label='secondary catalog',
                zorder=secondary_zorder)
        pl.plot(secondaryCat.ra[idx_secondaryCat], secondaryCat.dec[idx_secondaryCat], 'ko', label='xmatch sources',
                zorder=-20)
        ax = pl.gca()
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        pl.xlabel('RA (deg)')
        pl.ylabel('Dec (deg)')
        pl.legend()
        pl.show()
        if saveplot == 1:
            figName = os.path.join(outDir, '%s_xmatch_onSky.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

    # define quantities for quality check and further processing
    cosDecFactor = np.cos(np.deg2rad(primaryCat.dec[idx_primaryCat]))
    diff_raStar = (secondaryCat.ra[idx_secondaryCat] - primaryCat.ra[idx_primaryCat]) * cosDecFactor
    diff_de = (secondaryCat.dec[idx_secondaryCat] - primaryCat.dec[idx_primaryCat])

    xmatchDistance = d2d.copy().to(u.milliarcsecond)

    # display actual distortion
    if verboseFigures:
        X = primaryCat.ra[idx_primaryCat];
        Y = primaryCat.dec[idx_primaryCat];
        #         zeroPointIndex = np.where(diff_raStar == np.median(diff_raStar))[0][0]
        #         zeroPointIndex = np.where(diff_de == np.median(diff_de))[0][0]
        U0 = diff_raStar;
        V0 = diff_de;
        U = U0 - np.median(U0);
        V = V0 - np.median(V0);

        UV_factor = primaryCat.ra.unit.to(u.milliarcsecond)

        n_bins = np.int(len(xmatchDistance) / 5)

        # xmatch diagnostics
        pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k');
        pl.clf();
        pl.subplot(1, 2, 1)
        pl.hist(xmatchDistance, n_bins)
        pl.xlabel('Crossmatch distance (%s)' % (xmatchDistance.unit))
        pl.subplot(1, 2, 2)
        pl.hist(U0 * UV_factor, n_bins, color='b', label='X')
        pl.hist(V0 * UV_factor, n_bins, color='r', alpha=0.5, label='Y')
        pl.xlabel('Coordinate Difference in X and Y {}'.format(u.milliarcsecond))
        pl.legend(loc='best')
        fig.tight_layout(h_pad=0.0)
        pl.show()
        if saveplot == 1:
            figName = os.path.join(outDir, '%s_xmatch_distance.pdf' % nameSeed)
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
            DistanceModulus = np.abs(primaryCat[idx_primaryCat].separation(secondaryCat[idx_secondaryCat]))

            print('naiveDistanceModulus  : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(
                np.min(naiveDistanceModulus).to(u.milliarcsecond), np.max(naiveDistanceModulus).to(u.milliarcsecond),
                np.median(naiveDistanceModulus).to(u.milliarcsecond)))
            print('     DistanceModulus  : min {0:3.1f} max {1:3.1f} median {2:3.1f}'.format(
                np.min(DistanceModulus).to(u.milliarcsecond), np.max(DistanceModulus).to(u.milliarcsecond),
                np.median(DistanceModulus).to(u.milliarcsecond)))

            #         DistanceModulus -= np.median(DistanceModulus)
            #         naiveDistanceModulus = np.sqrt(U0**2+V0**2) - np.median(np.sqrt(U0**2+V0**2))

            pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k');
            pl.clf();
            pl.hist(U.to(u.milliarcsecond), 100, color='r')
            pl.hist(V.to(u.milliarcsecond), 100, color='b')
            pl.show()

            pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k');
            pl.clf();
            pl.plot(DistanceModulus.to(u.milliarcsecond), naiveDistanceModulus.to(u.milliarcsecond), 'b.')
            pl.show()

        fig = pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k');
        pl.clf();

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

        if saveplot == 1:
            figName = os.path.join(outDir, '%s_xmatch_distortionActual.pdf' % nameSeed)
            pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

    return idx_primaryCat, idx_secondaryCat, d2d, d3d, diff_raStar, diff_de
