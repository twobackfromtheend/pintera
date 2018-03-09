import data_io as dio
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sps
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

    def preprocess_xy(self, moving_y_offset_wavelengths=None):
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
        self.config['y_offset'] = str(self.y_offset)
        self.x = x
        self.y = norm_y

        if moving_y_offset_wavelengths is not None:
            self.y = self.y - self.get_moving_y_offset(moving_y_offset_wavelengths)

        return x, norm_y

    def get_moving_y_offset(self, wavelengths=5):
        max_x, max_y = self.get_local_maxes()
        average_steps_between_max = np.mean(np.ediff1d(max_x))

        y_offsets = np.zeros_like(self.y)

        steps_buffer = int(wavelengths * average_steps_between_max)

        # print(wavelengths * average_steps_between_max)
        # _i_start = np.argmax(self.x > self.x[0] + int(wavelengths * average_steps_between_max))
        # _i_end = np.argmax(self.x > self.x[-1] - int(wavelengths * average_steps_between_max))
        # print(np.argmax(self.x > max_x[wavelengths]), np.argmax(self.x > max_x[-wavelengths - 1]))
        # print(_i_start, _i_end)
        # for i in range(_i_start, _i_end + 1):
        #     _x, _y = self.x[i], self.y[i]
        #
        #     _x_i_start = np.argmax(self.x > _x -steps_buffer)
        #     _x_i_end = np.argmax(self.x > _x + steps_buffer)
        #
        #     y_offsets[i] = np.mean(self.y[_x_i_start:_x_i_end + 1])

        for i in range(steps_buffer, len(self.x) - steps_buffer):
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
        """Returns amplitude, mean, and sigma^2. Uses get_local_maxes to get x and y unless x_y is passed"""
        if x_y is None:
            x, y = self.get_local_maxes()
        else:
            x, y = x_y
        y_max = np.max(y)
        amplitude = np.max(y)

        mean = np.sum(x * y) / np.sum(y)
        sigma2 = np.abs(np.sum(y * (x - mean) ** 2) / np.sum(y))

        if also_use_scipy:
            if fix_mean is not False:
                # TODO: Allow fixed mean optimisation
                pass
            else:
                # fit_params is (amplitude, mean, sigma2)
                gaussian_fit = lambda fit_params, x: fit_params[0] * np.exp(-(x - fit_params[1]) ** 2 / (2 * fit_params[2]))
                err_func = lambda fit_params, x, y: gaussian_fit(fit_params, x) - y  # Distance to the target function
                initial_parameters = [y_max, mean, sigma2]  # Initial guess for the parameters
                fitted_params, success = optimize.leastsq(err_func, initial_parameters[:], args=(x, y))
                # print(fitted_params, success)

                if save:
                    self.config['gaussian_fit_amplitude'] = str(fitted_params[0])
                    self.config['gaussian_fit_mean'] = str(fitted_params[1])
                    self.config['gaussian_fit_sigma^2'] = str(fitted_params[2])

                return fitted_params, (amplitude, mean, sigma2)

        else:
            # calculate gaussian fit from points
            return amplitude, mean, sigma2

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
        # returns a string that can be pasted into
        _s = io.BytesIO()
        data = np.column_stack(args)
        np.savetxt(_s, data, fmt=('%i', '%.18e'), delimiter='\t')
        return _s.getvalue().decode()

if __name__ == '__main__':
    data_dir = 'data'  # change to blank string ('') if data is not in its own directory
    file_names = ['20S3', 'iP10S1', '10S2', 'W1S2', 'OPW5S1']

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
    signal_plotter.plot_motor_step_size_dps(signal)
    # find_step_size(x, y)
    #
    # plt.plot(x, y)
    #
    # max_x, max_y = signal.get_local_maxes()
    # print(max_x, max_y)
    #
    #
    # plt.plot(max_x, max_y, 'r.')
    # plt.show()
