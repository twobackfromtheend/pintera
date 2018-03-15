import data_io as dio
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sps
from scipy import constants
from scipy import optimize
import config


def create_signals(file_names, data_dir=''):
    signals = []
    for file_name in file_names:
        signal = Signal(file_name, data_dir)
        signals.append(signal)
    return signals


class Signal:
    base_config = config.load_config('params.cfg')

    def __init__(self, file_name, data_dir=''):
        # Contains data about the signal, and loads its config
        self.file_name = file_name
        self.file_path = os.path.join(data_dir, file_name)

        # parse signal first - to get errors before config is created.
        self.x_full, self.y_full, self.delta = dio.get_x_y_delta(self.file_path, v=False)
        # create config if not created
        if self.file_name not in self.base_config:
            self.base_config.add_section(self.file_name)
        self.config = self.base_config[self.file_name]

        # process data
        self.x, self.y = self.preprocess_xy()

        self.base_config.write()

    def preprocess_xy(self, moving_y_offset_wavelengths=None, use_abs=False):
        """Returns preprocessed x and y, based on the config"""
        x, y = self.x_full, self.y_full
        # remove boundary signals
        x_limits = self.get_x_lims()

        left_lim, right_lim = np.argmax(x > x_limits[0]), np.argmax(x > x_limits[1])
        if right_lim == 0:
            self.set_x_lims((x_limits[0], -1))
            print("Error: steps_lim_high > last step. Set to -1.")
        x = x[left_lim:right_lim]
        y = y[left_lim:right_lim]

        # shift to around x axis
        self.y_offset = np.mean(y)
        norm_y = y - self.y_offset
        if use_abs:
            norm_y = np.fabs(norm_y)
        self.config['y_offset'] = str(self.y_offset)
        self.x = x
        self.y = norm_y

        if moving_y_offset_wavelengths is not None:
            self.y = self.y - self.get_moving_y_offset(moving_y_offset_wavelengths)

        return x, norm_y

    def get_moving_y_offset(self, wavelengths=5):
        """
        Returns a NumPy array that can be used as a y-offset (e.g. y = y - y_offset).
        Averages out the prior and next n wavelengths, where n is the passed parameter.
        The distance of a wavelength is calculated as a simple mean distance between peaks).
        """
        max_x, max_y = self.get_local_maxes()
        average_steps_between_max = np.mean(np.ediff1d(max_x))

        y_offsets = np.zeros_like(self.y)

        steps_buffer = int(wavelengths * average_steps_between_max)

        for i in range(steps_buffer, len(self.x) - steps_buffer):
            # TODO: Optimise (probably can be done with NumPy)
            _x, _y = self.x[i], self.y[i]
            _steps_buffer = min(i, len(self.x) - i, steps_buffer)
            _x_i_start = np.argmax(self.x > _x - _steps_buffer)
            _x_i_end = np.argmax(self.x > _x + _steps_buffer)

            y_offsets[i] = np.mean(self.y[_x_i_start:_x_i_end + 1])

        return y_offsets

    def get_x_lims(self):
        steps_lim_low = self.config.getint('steps_lim_low')
        steps_lim_high = self.config.getint('steps_lim_high')
        if steps_lim_high == -1:
            steps_lim_high = max(self.x_full) - 1

        return steps_lim_low, steps_lim_high

    def set_x_lims(self, x_lims):
        self.config['steps_lim_low'] = str(x_lims[0])
        self.config['steps_lim_high'] = str(x_lims[1])
        self.base_config.write()

    def get_x_centre(self):
        x_centre = self.config.getint('x_centre')
        return x_centre

    def set_x_centre(self, x_centre):
        self.config['x_centre'] = str(x_centre)
        self.base_config.write()

    def get_y_offset(self):
        y_offset = self.config.getfloat('y_offset')
        if y_offset == 0:
            y_offset = np.mean(self.y)
        return y_offset

    def set_y_offset(self, y_offset):
        self.config['y_offset'] = str(y_offset)
        self.base_config.write()

    def get_local_maxes(self, use_full=False, strict=False, x_y=None):
        """Passing x_y uses those. Else signal's .x and .y attributes will be used (unless use_full is True)"""
        if x_y is None:
            if use_full:
                x, y = self.x_full, self.y_full
                y_offset = 0
            else:
                x, y = self.x, self.y
                y_offset = self.y_offset
        else:
            x, y = x_y
            y_offset = 0

        if strict:
            # take only those greater than both adjacent
            maxes = sps.argrelextrema(y, np.greater)[0]
        else:
            # take all greater/equal to both sides
            maxes = sps.argrelextrema(y, np.greater_equal)[0]

        # check that max_y values > 0
        maxes = maxes[y[maxes] > 0]

        # filter capped values on both sides
        maxes = maxes[y[maxes] != 5 - y_offset]

        max_x = x[maxes]
        max_y = y[maxes]

        return max_x, max_y

    def find_best_fit_gaussian(self, also_use_scipy=True, save=True, fix_mean=False, x_y=None):
        """
        Returns scipy_fit, calculated_fit
            each fit = (amplitude, mean, and sigma).

        SciPy fit generated using scipy.optimise, optimising all of amplitude, mean, and sigma.
        Calculated fit calculates the mean and sigma using formulae, and takes the max(y) as amplitude.

        Uses get_local_maxes to get x and y unless x_y is passed.
        TODO: Create fix_mean
        """
        if x_y is None:
            x, y = self.get_local_maxes()
        else:
            x, y = x_y
        y_max = np.max(y)
        amplitude = np.max(y)
        mean = np.sum(x * y) / np.sum(y)
        sigma = np.sqrt(np.abs(np.sum(y * (x - mean) ** 2) / np.sum(y)))

        if also_use_scipy:
            if fix_mean is not False:
                # TODO: Allow fixed mean optimisation
                pass
            else:
                # fit_params is (amplitude, mean, sigma2)
                gaussian_fit = lambda fit_params, x: fit_params[0] * np.exp(
                    -(x - fit_params[1]) ** 2 / (2 * fit_params[2] ** 2))
                err_func = lambda fit_params, x, y: gaussian_fit(fit_params, x) - y  # Distance to the target function
                initial_parameters = [y_max, mean, sigma]  # Initial guess for the parameters
                fitted_params, success = optimize.leastsq(err_func, initial_parameters[:], args=(x, y))
                # print(fitted_params, success)

                if save:
                    self.config['gaussian_fit_amplitude'] = str(fitted_params[0])
                    self.config['gaussian_fit_mean'] = str(fitted_params[1])
                    self.config['gaussian_fit_sigma'] = str(fitted_params[2])

                return fitted_params, (amplitude, mean, sigma)

        else:
            # calculate gaussian fit from points
            return amplitude, mean, sigma

    def find_best_fit_lorentzian(self, save=True, fix_mean=False, x_y=None):
        """
        Returns scipy_fit: (amplitude, mean, and gamma).

        SciPy fit generated using scipy.optimise, optimising all of amplitude, mean, and gamma.
        Calculated fit calculates the mean and sigma^2 using formulae, and takes the max(y) as amplitude.

        Uses get_local_maxes to get x and y unless x_y is passed.
        TODO: Create fix_mean
        """
        if x_y is None:
            x, y = self.get_local_maxes()
        else:
            x, y = x_y
        y_max = np.max(y)

        mean = np.sum(x * y) / np.sum(y)
        sigma = np.sqrt(np.abs(np.sum(y * (x - mean) ** 2) / np.sum(y)))

        if fix_mean is not False:
            # TODO: Allow fixed mean optimisation
            pass
        else:
            # fit_params is (amplitude, mean, gamma)
            lorentzian_fit = lambda fit_params, x: fit_params[0] / (1 + ((x - fit_params[1]) / fit_params[2]) ** 2)
            err_func = lambda fit_params, x, y: lorentzian_fit(fit_params, x) - y  # Distance to the target function
            initial_parameters = [y_max, mean, sigma]  # Initial guess for the parameters
            fitted_params, success = optimize.leastsq(err_func, initial_parameters[:], args=(x, y))
            if save:
                self.config['lorentzian_fit_amplitude'] = str(fitted_params[0])
                self.config['lorentzian_fit_mean'] = str(fitted_params[1])
                self.config['lorentzian_fit_gamma'] = str(fitted_params[2])

            return fitted_params

    def find_best_fit_exponential(self, save=True, x_y=None, beating=True):
        """
        Returns scipy_fit: (amplitude, mean, decay_constant, [beating_freq]). (beating_freq

        SciPy fit generated using scipy.optimise, optimising all of amplitude, mean, and gamma.
        Calculated fit calculates the mean and sigma^2 using formulae, and takes the max(y) as amplitude.

        Uses get_local_maxes to get x and y unless x_y is passed.
        """
        if x_y is None:
            x, y = self.get_local_maxes()
        else:
            x, y = x_y
        y_max = np.max(y)

        mean = np.sum(x * y) / np.sum(y)
        sigma = np.sqrt(np.abs(np.sum(y * (x - mean) ** 2) / np.sum(y)))

        if beating:
            # fit_params is (amplitude, mean, decay_constant, beating_freq)
            exp_fit = lambda fit_params, x: fit_params[0] * np.exp(-fit_params[2] * np.fabs(x - fit_params[1])) * \
                                            np.fabs(np.cos(2 * np.pi * (x - fit_params[1]) * fit_params[3]))
            err_func = lambda fit_params, x, y: exp_fit(fit_params, x) - y  # Distance to the target function
            initial_parameters = [y_max, mean, 1 / sigma, 1 / (4 * sigma)]
            fitted_params, success = optimize.leastsq(err_func, initial_parameters[:], args=(x, y))
            if save:
                self.config['exponential_fit_amplitude'] = str(fitted_params[0])
                self.config['exponential_fit_mean'] = str(fitted_params[1])
                self.config['exponential_fit_decay_constant'] = str(fitted_params[2])
                self.config['exponential_fit_beating_freq'] = str(fitted_params[3])

        else:
            # fit_params is (amplitude, mean, decay_constant)
            exp_fit = lambda fit_params, x: fit_params[0] * np.exp(-fit_params[2] * np.fabs(x - fit_params[1]))
            err_func = lambda fit_params, x, y: exp_fit(fit_params, x) - y  # Distance to the target function
            initial_parameters = [y_max, mean, 1 / sigma]
            fitted_params, success = optimize.leastsq(err_func, initial_parameters[:], args=(x, y))
            if save:
                self.config['exponential_fit_amplitude'] = str(fitted_params[0])
                self.config['exponential_fit_mean'] = str(fitted_params[1])
                self.config['exponential_fit_decay_constant'] = str(fitted_params[2])

        return fitted_params

    def find_step_size(self, known_wavelength=546.1E-9, bins=1):
        # dt = lambda / 2
        # displacement per step = peaks / steps * lambda / 2
        bin_width = int(np.floor(len(self.x) / bins))
        peaks_list = []
        steps_list = []

        # all remainder is ignored (e.g. data 1000 to 1050 if 1050 split into 100 bins)
        for bin_i in range(bins):
            offset = bin_width * bin_i
            right_lim = offset + bin_width - 1
            # print(offset, right_lim)
            max_x, max_y = self.get_local_maxes(x_y=(self.x[offset:right_lim], self.y[offset:right_lim]))
            _peaks = len(max_x)
            _steps = self.x[right_lim] - self.x[offset]
            peaks_list.append(_peaks)
            steps_list.append(_steps)
        # print(peaks_list, steps_list)
        steps_list = np.array(steps_list)
        peaks_list = np.array(peaks_list)
        dps = peaks_list / steps_list * known_wavelength / 2
        # print('DPS: %.4e, pm %.1e '% (dps.mean(), np.std(dps)))

        return dps, bin_width

    @staticmethod
    def as_string(*args):
        # returns a string that can be pasted into Excel/Origin/other stuff (tab as delimiter)
        _s = io.BytesIO()
        data = np.column_stack(args)
        np.savetxt(_s, data, fmt=('%i', '%.18e'), delimiter='\t')
        return _s.getvalue().decode()

    def get_steps_between_peaks(self):
        """
        Returns steps_data and dps_data, where
            steps_data = (steps, unique_steps_between_peaks, unique_steps_counts)

        Calculates the number of steps between each peak and the next
        (e.g. if 5 peaks found, there will be 4 "steps between peaks", and the sum of unique steps_count will be 4)
        unique_steps_between_peaks is an array of multiples of self.delta.

        Filters out steps between peaks that are < 0.3 or > 1.7 times the modal steps count.
        1.7 chosen to avoid the small second peak at 2 times (probably due to single missed peaks), and
        0.3 chosen to preserve symmetry about the peak.
        """
        max_x, max_y = self.get_local_maxes()
        full_steps = np.ediff1d(max_x)
        # _full_mean, _full_std = np.mean(full_steps), np.std(full_steps)
        _full_count = len(full_steps)

        unique_steps_between_peaks, unique_steps_counts = np.unique(full_steps, return_counts=True)

        _filter = np.logical_and(full_steps < unique_steps_between_peaks[np.argmax(unique_steps_counts)] * 1.7,
                                 full_steps > unique_steps_between_peaks[np.argmax(unique_steps_counts)] * 0.3)
        # 1.7 chosen as filter, as there seems to be another peak ~2* (probably due to single missed peaks)
        # 1.7 avoids the start of the gaussian at 2*

        if not _filter.all():
            steps = full_steps[_filter]
            # print(unique_steps_between_peaks[np.argmax(unique_steps_counts)])
            _filtered_count = len(steps)
            _counts = (_full_count, _filtered_count, _full_count - _filtered_count)
            # print('Original Count: %s, Filtered Count: %s, Excluded Count: %s' % _counts)
            # print('Filtered:', full_steps[np.invert(_filter)])
            unique_steps_between_peaks, unique_steps_counts = np.unique(steps, return_counts=True)
        else:
            steps = full_steps

        return steps, unique_steps_between_peaks, unique_steps_counts

    def get_motor_step_dps_per_peak(self, known_wavelength):
        """
        Takes in a known wavelength (in metres) to find motor step size (calibration).
        Returns steps_data and dps_data, where
            steps_data = (steps, unique_steps_between_peaks, unique_steps_counts)
            dps_data = (_dpses, unique_dpses, unique_dpses_counts, dps_mean, dps_std)

        Finds the number of steps between maxima, then finds the corresponding displacement per step (dps).
        If there are 5 maxima points, there will be 4 data points (4 'steps between maxima' to 4 'dps').
        Returns the mean and std for dps. Other returned information (such as steps_data) is returned for plotting use.
        """
        steps, unique_steps_between_peaks, unique_steps_counts = self.get_steps_between_peaks()

        _dpses = known_wavelength / (2 * steps)
        dps_mean, dps_std = np.mean(_dpses), np.std(_dpses)
        unique_dpses, unique_dpses_counts = np.unique(_dpses, return_counts=True)
        print('DPS: %s, DPS std dev: %s' % (dps_mean, dps_std))

        steps_data = (steps, unique_steps_between_peaks, unique_steps_counts)
        dps_data = (_dpses, unique_dpses, unique_dpses_counts, dps_mean, dps_std)
        return steps_data, dps_data

    def get_motor_step_dps_with_fourier(self, known_wavelength, freq_limits=(2e-3, 1.5e-2)):
        fourier = np.fft.fft(self.y)
        freqs_full = np.fft.fftfreq(self.y.size, d=self.delta)

        freq_filter = np.where(np.logical_and(freqs_full >= freq_limits[0], freqs_full <= freq_limits[1]))
        frequencies = freqs_full[freq_filter]
        magnitudes = abs(fourier[freq_filter])

        # steps between peaks = 1 wavelength = 1 / freq
        # dps = known_wavelength / (2 * 1 wavelength) = freq * known_wavelength / 2
        _dpses = frequencies * known_wavelength / 2

        scipy_fit, calc_fit = self.find_best_fit_gaussian(x_y=(_dpses, magnitudes))
        print('DPS: Mean: %.4e, std: %.4e' % (scipy_fit[1], scipy_fit[2]))

        return _dpses, scipy_fit, frequencies, magnitudes

    def get_investigation_data(self, gamma, dps, gamma_err=0, dps_err=0, fit_type='exponential'):
        """Pass the standard deviation of the gaussian fit in terms of motor_steps
        and the displacement per motor step"""
        if fit_type == 'exponential':
            coherence_length = 2 * np.log(2) / (gamma / dps)

            # fourier transform of an exponential decay with decay constant g = lorentzian with hwhm g
            # decay constant = 1/g
            # Fourier(exp(-2pi k0 x)) = (1/pi)(k0 / (k^2 + k0^2))
            # 2 pi k0 = g, k0 = hwhm = g / (2 pi)
            # fwhm = g / pi (*c)
            spectral_width_hz = (gamma / dps) / np.pi * constants.c

        elif fit_type == 'lorentzian':
            coherence_length_in_motor_steps = 2 * gamma
            coherence_length = coherence_length_in_motor_steps * dps  # in metres

            spectral_width_hz = constants.c / (np.pi * coherence_length)

        elif fit_type == 'gaussian':
            # gamma is actually sigma
            coherence_length_in_motor_steps = 2 * np.sqrt(2 * np.log(2)) * gamma
            coherence_length = coherence_length_in_motor_steps * dps  # in metres

            spectral_width_hz = constants.c / (np.pi * coherence_length)

        steps, unique_steps_between_peaks, unique_steps_counts = self.get_steps_between_peaks()

        distances = dps * steps
        # distances_mean, distances_std = np.mean(distances), np.std(distances)
        wavelengths = distances * 2
        wavelengths_mean, wavelengths_std = np.mean(wavelengths), np.std(wavelengths)

        mean_wavelength = wavelengths_mean

        frequencies = constants.c / wavelengths
        frequencies_mean, frequencies_std = np.mean(frequencies), np.std(frequencies)

        spectral_width_m = mean_wavelength ** 2 / constants.c * spectral_width_hz
        print('spec_width (Hz): %.5e' % spectral_width_hz)
        print('spec_width (m): %.5e' % spectral_width_m)

        if gamma_err == 0 and dps_err == 0:
            print('Coherence length: %.5e' % coherence_length)
            print('Spectral width: %.5e' % spectral_width_hz)
            print('Mean frequencies: %.5e pm %.5e' % (frequencies_mean, frequencies_std))
            print('Mean wavelength: %.5e pm %.5e' % (mean_wavelength, wavelengths_std))
        else:
            print(dps_err, 'asdasdasfaegsegeg')
            coherence_length_err = dps_err / dps * coherence_length
            spectral_width_err = np.sqrt((coherence_length_err / (constants.c * mean_wavelength ** 2)) ** 2 +
                                         (2 * coherence_length / (constants.c * mean_wavelength ** 3)) ** 2)
            print(mean_wavelength ** 2 / coherence_length, 'dlambda')
            print('Coherence length: %.5e pm %.5e' % (coherence_length, coherence_length_err))
            print('Spectral width: %.5e pm %.5e' % (spectral_width_hz, spectral_width_err))
            print('Mean frequencies: %.5e pm %.5e' % (frequencies_mean, frequencies_std))
            print('Mean wavelength: %.5e pm %.5e' % (mean_wavelength, wavelengths_std))

        data = {'coherence_length': coherence_length,
                'spectral_width_hz': spectral_width_hz,
                'spectral_width_m': spectral_width_m,
                'mean_wavelength': mean_wavelength,
                'mean_frequency': frequencies_mean,
                }
        return data


if __name__ == '__main__':
    data_dir = 'data'  # change to blank string ('') if data is not in its own directory
    file_names = ['20S3', 'iP10S1', '10S2', 'W1S2', 'W10S1', 'OPW5S1']

    signals = create_signals(file_names, data_dir)

    signal = signals[0]

    # plt.plot(signal.x, signal.y)
    # plt.show()
    # signal.plot_step_size()

    signal.find_best_fit_gaussian()
    # plt.show()
    # signal.as_string(*signal.get_local_maxes())

    # _y_offset = signal.get_moving_y_offset()
    # plt.plot(signal.x, signal.y - _y_offset, 'r', alpha=0.3)

    import signal_plotter

    signal_plotter.plot_motor_step_dps_with_fourier(signal)

    # signal.get_motor_step_dps_with_fourier(known_wavelength=546.1e-9)
    # signal.find_best_fit_gaussian(x_y=)
