"""Functions to facilitate crossmatching of sky catalogs.

Authors
-------
    Johannes Sahlmann


References
----------
    This module is build on top of the ``astropy.coordinates`` package and its `search_around_sky`
    method.

"""
import os

import numpy as np
import pylab as pl
from matplotlib.ticker import FormatStrFormatter
import astropy.units as u


def xmatch(primary_cat, secondary_cat, xmatch_radius, rejection_level_sigma=0,
           remove_multiple_matches=True, retain_best_match=False, verbose=False,
           verbose_figures=False, save_plot=False, out_dir=None, name_seed=None):
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
    save_plot : bool
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
        if save_plot:
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
        if save_plot:
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

        if save_plot:
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
