# import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'

current_patches = []


def plot_signal(signal, ax, fig=None, plot_style='x-', plot_max=True, show_full_y=False, toolbar=None, plot_fit=True,
                dps=None):
    if ax is None:
        pass
        # ax = plt.gca()

    # print(ax)
    ax.clear()
    if dps is None:
        ax.plot(signal.x_full, signal.y_full - signal.y_offset, plot_style, markersize=2, alpha=0.3, label='Full')
        ax.plot(signal.x, signal.y, 'r', alpha=0.5, label='Preprocessed')
    else:
        ax.plot(signal.x_full * dps, signal.y_full - signal.y_offset, plot_style, markersize=2, alpha=0.3,
                label='Full')
        ax.plot(signal.x * dps, signal.y, 'r', alpha=0.5, label='Preprocessed')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if show_full_y:
        y_min = min(y_min, -2.5)
        y_max = max(y_max, 2.5)
    width = x_max - x_min
    height = y_max - y_min

    # top_patch = Rectangle((x_min, 5 - signal.y_offset), width, y_max - (5 - signal.y_offset), alpha=0.3, color='grey', lw=0)
    # patch can start from the top with negative height
    top_patch = Rectangle((x_min, y_max), width, 5 - signal.y_offset - y_max, alpha=0.3, color='grey', lw=0)
    # print(0 - signal.y_offset - height)
    bottom_patch = Rectangle((x_min, y_min), width, - y_min - signal.y_offset, alpha=0.3, color='grey', lw=0)

    signal_x_lims = signal.get_x_lims()
    if dps is not None:
        signal_x_lims = [x * dps for x in signal_x_lims]
    left_patch = Rectangle((x_min, y_min), signal_x_lims[0] - x_min, height, alpha=0.3,
                           color='lightgrey', lw=0)
    # right_patch = Rectangle((signal_x_lims[1], y_min), x_max - signal_x_lims[1], height, alpha=0.3, color='lightgrey', lw=0)
    # patch can start from right with negative width
    right_patch = Rectangle((x_max, y_min), signal_x_lims[1] - x_max, height, alpha=0.3, color='lightgrey', lw=0)
    global current_patches
    current_patches = [top_patch, bottom_patch, left_patch, right_patch]
    ax.add_patch(top_patch)
    ax.add_patch(bottom_patch)
    ax.add_patch(left_patch)
    ax.add_patch(right_patch)

    if plot_max:
        max_x, max_y = signal.get_local_maxes()
        if dps is None:
            ax.plot(max_x, max_y, 'r.', label='maxima')
        else:
            ax.plot(max_x * dps, max_y, 'r.', label='maxima')
    if plot_fit:
        scipy_fit, calc_fit = plot_best_fit_gaussian(signal, ax, dps=dps)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    if dps is None:
        ax.set_xlabel('Motor Steps')
        ax.set_ylabel('Adjusted Intensity')
    else:
        ax.set_xlabel('Displacement (m)')
        ax.set_ylabel('Adjusted Intensity')

    ax.legend()
    ax.grid()

    if toolbar is not None:
        toolbar.update()

    if fig is not None:
        fig.tight_layout()

    if plot_fit:
        return scipy_fit, calc_fit


def update_patches(signal, ax, dps=None):
    signal_x_lims = signal.get_x_lims()

    x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()
    if dps is None:
        # left patch
        current_patches[2].set_width(signal_x_lims[0] - x_min)
        # right patch
        current_patches[3].set_width(signal_x_lims[1] - x_max)
    else:
        # left patch
        current_patches[2].set_width(signal_x_lims[0] * dps - x_min)
        # right patch
        current_patches[3].set_width(signal_x_lims[1] * dps - x_max)


def plot_best_fit_gaussian(signal, ax, plot_scipy=True, plot_calc=False, dps=None):
    x_min, x_max = ax.get_xlim()
    scipy_fit, calc_fit = signal.find_best_fit_gaussian(also_use_scipy=True)
    print('Fits:  \tScipy: \t\t\t%.5e, \t%.5e, \t%.5e,\n\t\tCalculated: \t%.5e, \t%.5e, \t%.5e' % (*scipy_fit, *calc_fit))

    if dps is not None:
        calc_fit = calc_fit[0], calc_fit[1] * dps, calc_fit[2] * dps**2
        scipy_fit = scipy_fit[0], scipy_fit[1] * dps, scipy_fit[2] * dps ** 2

    # fit_params is (amplitude, mean, sigma2)
    gaussian_fit = lambda fit_params, x: fit_params[0] * np.exp(-(x - fit_params[1]) ** 2 / (2 * fit_params[2]))

    if plot_scipy:
        # plot scipy
        fit_x = np.linspace(x_min, x_max, 10000)
        optimised_gaussian_fit = gaussian_fit(scipy_fit, fit_x)
        ax.plot(fit_x, optimised_gaussian_fit, 'k', label='SciPy fit')
    if plot_calc:
        # plot calculated fit
        fit_x = np.linspace(x_min, x_max, 10000)
        calculated_gaussian_fit = gaussian_fit(calc_fit, fit_x)
        ax.plot(fit_x, calculated_gaussian_fit, 'y', label='calc. fit')

    return scipy_fit, calc_fit


def plot_motor_step_dps_with_bins(signal, known_wavelength=None):
    max_x, max_y = signal.get_local_maxes()
    maxima_count = len(max_x)

    bins_list = np.linspace(1, int(maxima_count / 5), int(maxima_count / 5), dtype=np.uint32)
    dps_mean_list = []
    dps_err_list = []

    for bins in bins_list:
        if known_wavelength:
            dps, bin_width = signal.find_step_size(known_wavelength=known_wavelength, bins=bins)
        else:
            dps, bin_width = signal.find_step_size(bins=bins)
        dps_mean = np.mean(dps)
        dps_err = np.std(dps)
        dps_mean_list.append(dps_mean)
        dps_err_list.append(dps_err)
    # print(bins_list, dps_mean_list, dps_err_list)
    plt.errorbar(bins_list, dps_mean_list, yerr=dps_err_list, fmt='.')
    plt.title('Displacement per step for varying number of bins')
    plt.ylabel('Displacement per step (m)')
    plt.xlabel('Number of bins')

    _dps_mean_list = []
    _dps_err_list = []
    for bins in range(5, 10 + 1):
        dps, bin_width = signal.find_step_size(bins=bins)
        dps_mean = np.mean(dps)
        dps_err = np.std(dps)
        _dps_mean_list.append(dps_mean)
        _dps_err_list.append(dps_err)
    dps_5_10_mean = np.mean(_dps_mean_list)
    dps_5_10_err = np.mean(_dps_err_list)
    print("Average DPS for 5-10 bins: %.4e pm %.1e" % (dps_5_10_mean, dps_5_10_err))
    plt.errorbar(7.5, dps_5_10_mean, xerr=3, yerr=dps_5_10_err, fmt='r.')
    plt.show()

    return dps_5_10_mean, dps_5_10_err


def plot_motor_step_size_dps_per_peak(signal, known_wavelength=546.1e-9):
    steps_data, dps_data = signal.get_motor_step_size_dps_per_peak(known_wavelength)
    steps, unique_steps_between_peaks, unique_steps_counts = steps_data
    _dpses, unique_dpses, unique_dpses_counts, dps_mean, dps_std = dps_data
    fig = plt.figure(figsize=(10, 6))

    # Plot 1:
    ax1 = fig.add_subplot(121)
    # plot motor steps between peaks
    # fit_params is (area, mean, sigma2)
    gaussian_fit = lambda fit_params, x: fit_params[0] / np.sqrt(2 * np.pi * fit_params[2] ** 2) * np.exp(
        -(x - fit_params[1]) ** 2 / (2 * fit_params[2] ** 2))

    plt.plot(unique_steps_between_peaks, unique_steps_counts, '.')
    x_min, x_max = plt.gca().get_xlim()

    # plot scipy
    fit_x = np.linspace(x_min, x_max, 10000)
    fit_area = len(steps) * signal.delta
    steps_between_peaks_mean, steps_between_peaks_std = np.mean(steps), np.std(steps)

    calculated_fit = gaussian_fit((fit_area, steps_between_peaks_mean, steps_between_peaks_std), fit_x)
    ax1.plot(fit_x, calculated_fit, 'k', alpha=0.6,
             label=r'$\mu=$%.2e,%s$\sigma=$%.1e' % (steps_between_peaks_mean, '\n', steps_between_peaks_std))
    ax1.legend(loc=1)

    ax1.set_ylabel('Count of motor steps between maxima')
    ax1.set_xlabel('Motor steps between maxima')

    # Plot 2:
    ax2 = fig.add_subplot(122)
    # plot dps from each "motor step between peaks"
    gaussian_fit = lambda fit_params, x: fit_params[0] / np.sqrt(2 * np.pi * fit_params[2] ** 2) * np.exp(
        -(x - fit_params[1]) ** 2 / (2 * fit_params[2] ** 2))

    ax2.plot(unique_dpses, unique_dpses_counts, '.')
    x_min, x_max = plt.gca().get_xlim()

    # the following does not find mu/sigma. it only optimises the area so the fit looks good.
    from scipy import optimize
    gaussian_fit = lambda area, x: area / np.sqrt(2 * np.pi * dps_std ** 2) * np.exp(
        -(x - dps_mean) ** 2 / (2 * dps_std ** 2))
    err_func = lambda fit_params, x, y: gaussian_fit(fit_params, x) - y  # Distance to the target function
    initial_parameters = [len(steps) * known_wavelength / (2 * signal.delta)]  # Initial guess for the parameters

    fitted_area, success = optimize.leastsq(err_func, initial_parameters[:], args=(unique_dpses, unique_dpses_counts))

    # plot scipy
    fit_x = np.linspace(x_min, x_max, 10000)
    # I cannot find a way to deal with the non-linear x values (to find fit_area). Using scipy.optimise as cop out
    fit_area = fitted_area
    calculated_fit = gaussian_fit(fit_area, fit_x)
    ax2.plot(fit_x, calculated_fit, 'k', alpha=0.6, label=r'$\mu=$%.2e%s$\sigma=$%.1e' % (dps_mean, '\n', dps_std))

    ax2.legend(loc=1)

    ax2.set_ylabel('Count of Displacement per Step')
    ax2.set_xlabel('Displacement per step (m)')

    plt.tight_layout()
    plt.show()
