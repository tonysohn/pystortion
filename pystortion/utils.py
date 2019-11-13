import os

import numpy as np
import pylab as pl




def plot_spatial_difference(data, siaf=None, plot_aperture_names=None,
                            figure_types=['data', 'quiver', 'offset-corrected-quiver'],
                            make_new_figure=True, xy_label=['v2', 'v3'], xy_unit='arcsec',
                            plot_dir=None, name_seed='spatial_difference', title_string='',
                            quiver_scale=None, verbose=True, show_plot=True):
    """Make figures that show spatial x-y-data and their vector differences.

    Parameters
    ----------
    data
    siaf
    plot_aperture_names
    figure_types

    """
    reference_name = 'reference'

    if make_new_figure is False:
        figure_types = ['quiver']

    for figure_type in figure_types:
        if make_new_figure:
            pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
        if figure_type == 'data':
            pl.plot(data['reference']['x'], data['reference']['y'], 'b.', label='ref')
            pl.plot(data['comparison_0']['x'], data['comparison_0']['y'], 'r.', label='comp')
        else:
            for comparison_name in ['comparison_0']:
                delta_x = data[comparison_name]['x'] - data[reference_name]['x']
                delta_y = data[comparison_name]['y'] - data[reference_name]['y']

                if figure_type == 'offset-corrected-quiver':
                    delta_x -= np.mean(delta_x)
                    delta_y -= np.mean(delta_y)
                    title_string = 'Offset-subtracted '

                pl.quiver(data[reference_name]['x'], data[reference_name]['y'], delta_x, delta_y,
                          angles='xy', scale=quiver_scale)
                offsets = np.linalg.norm([delta_x, delta_y], axis=0)

                largest_difference = np.max(offsets)
                if verbose:
                    print('RMS of difference {:2.3f} mas'.format(np.std(offsets) * 1e3))
                    print('Maximum, minimum, PTV difference {:2.3f}, {:2.3f}, {:2.3f} mas'.format(np.max(offsets) * 1e3, np.min(offsets) * 1e3, np.ptp(offsets) * 1e3))
                pl.title('{}max. difference {:2.3f} mas'.format(title_string, largest_difference * 1e3))
        if plot_aperture_names is not None:
            for aperture_name in plot_aperture_names:
                siaf[aperture_name].plot()
        if make_new_figure:
            pl.axis('tight')
            pl.axis('equal')
            pl.xlabel('{} ({})'.format(xy_label[0], xy_unit))
            pl.ylabel('{} ({})'.format(xy_label[1], xy_unit))
            pl.legend(loc='best')
            ax = pl.gca()
            ax.invert_yaxis()
            if show_plot:
                pl.show()
            if plot_dir is not None:
                fig_name = os.path.join(plot_dir, '{}_{}.pdf'.format(name_seed, figure_type))
                pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0)

