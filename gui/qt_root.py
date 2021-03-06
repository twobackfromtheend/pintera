from gui.gui import Ui_MainWindow
from i_signal import Signal
import signal_plotter
import sys
import os

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
import matplotlib

matplotlib.use('QT5Agg')


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=10):
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, 'Select signal to plot.', va='center', ha='center', transform=self.fig.transFigure)
        self.ax.axis('off')
        super().__init__(self.fig)
        super().setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        super().updateGeometry()
        # Canvas.__init__(self, self.fig)
        # Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Canvas.updateGeometry(self)


class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

    # def resizeEvent(self, event):
    #     QtWidgets.QWidget.resizeEvent(self, event)
    #     print(self.geometry())
    #     self.canvas.draw()


class Analyser(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, base_config):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        # replace plot_widget with MplWidget
        self.plot_widget.hide()
        self.plot_widget = MplWidget(self.plot_frame)
        self.plot_widget.setMinimumSize(QtCore.QSize(0, 300))
        # self.addToolBar(NavigationToolbar(self.plot_widget.canvas, self), QtCore.Qt.ToolBarArea(1))
        self.toolbar = NavigationToolbar(self.plot_widget.canvas, self)
        self.addToolBar(QtCore.Qt.ToolBarArea(8), self.toolbar)

        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
        #                                    QtWidgets.QSizePolicy.MinimumExpanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        # self.plot_widget.setSizePolicy(sizePolicy)
        self.plot_widget.setObjectName("plot_widget")
        self.plot_layout.addWidget(self.plot_widget)

        # self.plot_data()
        self.connect_functions()
        self.data_file_names = []
        self.data_signals = []

        self.base_config = base_config
        self.import_signals()

    @staticmethod
    def main(base_config):
        app = QtWidgets.QApplication(sys.argv)
        window = Analyser(base_config)
        # window.load_presets_from_config(config, statusbar_update=False)
        window.show()
        app.exec_()

    def import_signals(self):
        data_dir = self.base_config['DEFAULT']['data_dir']
        for file_name in os.listdir(data_dir):
            if not file_name.endswith('.csv'):
                try:
                    signal = Signal(file_name, data_dir)
                    self.data_file_names.append(file_name)
                    self.data_signals.append(signal)
                except Exception as e:
                    print("Could not convert %s to signal. Error: %s" % (file_name, e))
        print("Created %s signals." % len(self.data_signals))

        self.signals_list_widget.clear()
        self.signals_list_widget.addItems(self.data_file_names)
        # if self.data_file_names:
        #     self.signals_list_widget.setCurrentRow(0)

    def connect_functions(self):
        self.signals_list_widget.itemSelectionChanged.connect(self.load_selected_signal)

        self.signal_options_reset_push_button.clicked.connect(self.reset_signal_options)
        for _spin_box in self.signal_options_x_frame.findChildren(QtWidgets.QSpinBox):
            # print(_spin_box.objectName())
            _spin_box.valueChanged.connect(self.signal_options_changed)

        # self.y_offset_double_spin_box.valueChanged.connect(self.signal_options_changed)
        # non-editable.

        self.recalculate_push_button.clicked.connect(self.recalculate)

        self.y_offset_moving_radio_button.toggled.connect(self.toggle_y_offset_type)
        self.exponential_radio_button.toggled.connect(self.toggle_fit_type)
        self.lorentzian_radio_button.toggled.connect(self.toggle_fit_type)
        self.gaussian_radio_button.toggled.connect(self.toggle_fit_type)
        self.fit_beat_check_box.toggled.connect(self.toggle_use_beating_freq)
        self.use_abs_checkbox.toggled.connect(self.recalculate)

        self.find_dps_push_button.clicked.connect(self.find_motor_dps_calibration)

        self.use_dist_as_x_checkbox.toggled.connect(self.toggle_x_dist)

        self.export_full_push_button.clicked.connect(self.export_to_clipboard)
        self.export_maxes_push_button.clicked.connect(self.export_to_clipboard)
        self.export_steps_push_button.clicked.connect(self.export_to_clipboard)
        self.export_ft_push_button.clicked.connect(self.export_to_clipboard)

    def load_selected_signal(self):
        try:
            _signal_name = self.signals_list_widget.currentItem().text()
            _signal_index = self.signals_list_widget.currentRow()
            # print(_signal_name, self.signals_list_widget.currentRow())

            print('\nLoading signal %s' % _signal_name)

            signal = self.data_signals[_signal_index]
            self.recalculate()

            x_lims = signal.get_x_lims()
            self.x_lim_spin_box_1.setValue(x_lims[0])
            self.x_lim_spin_box_2.setValue(x_lims[1])
            self.statusbar.showMessage(
                'Loaded signal %s.' % _signal_name, 10000)

            self.x_centre_spin_box.setValue(signal.get_x_centre())
            self.y_offset_double_spin_box.setValue(signal.get_y_offset())
        except:
            import traceback
            print("Error loading signal:")
            traceback.print_exc()

    def signal_options_changed(self):
        autosave = self.signal_options_autosave_check_box.isChecked()
        if autosave:
            _signal_name = self.signals_list_widget.currentItem().text()
            _signal_index = self.signals_list_widget.currentRow()
            # print(_signal_name, self.signals_list_widget.currentRow())
            signal = self.data_signals[_signal_index]

            # SAVE ALL THE OPTIONS
            x_lims = self.x_lim_spin_box_1.value(), self.x_lim_spin_box_2.value()
            signal.set_x_lims(x_lims)

            x_centre = self.x_centre_spin_box.value()

            signal.set_x_centre(x_centre)

            self.statusbar.showMessage(
                'Updated options for signal %s' % _signal_name)

            # self.draw_signal(signal)
            self.update_patches(signal)

    def reset_signal_options(self):
        _signal_name = self.signals_list_widget.currentItem().text()
        _signal_index = self.signals_list_widget.currentRow()
        # print(_signal_name, self.signals_list_widget.currentRow())
        signal = self.data_signals[_signal_index]

        signal.set_x_lims((0, -1))

        x_lims = signal.get_x_lims()
        self.x_lim_spin_box_1.setValue(x_lims[0])
        self.x_lim_spin_box_2.setValue(x_lims[1])
        self.statusbar.showMessage(
            'Reset options for signal %s' % _signal_name)

    def draw_signal(self, signal):
        if self.use_dist_as_x_checkbox.isChecked():
            dps = self.dps_spin_box.value() * 1e-9
        else:
            dps = None
        if self.lorentzian_radio_button.isChecked():
            fit_type = 'lorentzian'
        elif self.gaussian_radio_button.isChecked():
            fit_type = 'gaussian'
        elif self.exponential_radio_button.isChecked():
            fit_type = 'exponential'
        else:
            print('Neither lorentzian nor gaussian checked (in qt_root draw_signal)')

        separate_plot = self.separate_plot_checkbox.isChecked()
        if separate_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            scipy_fit = signal_plotter.plot_signal(signal, ax,
                                                   fig=fig,
                                                   toolbar=self.toolbar,
                                                   dps=dps,
                                                   fit_type=fit_type,
                                                   beating=self.fit_beat_check_box.isChecked())
            plt.show()
        else:
            scipy_fit = signal_plotter.plot_signal(signal, self.plot_widget.canvas.ax,
                                                   fig=self.plot_widget.canvas.fig,
                                                   toolbar=self.toolbar,
                                                   dps=dps,
                                                   fit_type=fit_type,
                                                   beating=self.fit_beat_check_box.isChecked())
            self.plot_widget.canvas.draw()

        # update spinboxes
        # fit
        sigma = scipy_fit[2]
        self.fit_a_double_spin_box.setValue(scipy_fit[0])
        self.fit_m_double_spin_box.setValue(scipy_fit[1])
        self.fit_s_double_spin_box.setValue(sigma)
        if self.fit_beat_check_box.isChecked():
            if len(scipy_fit) > 3:
                self.fit_beat_freq_double_spin_box.setValue(scipy_fit[3])
            else:
                print("SciPy fit does not contain beating freq (len=%s, not >3" % len(scipy_fit))
        # data
        if self.use_dist_as_x_checkbox.isChecked():
            # return sigma to in terms of motor steps if sigma is in terms of dps
            # get_investigation_data uses sigma in terms of motor steps
            if fit_type == 'exponential':
                sigma = sigma * dps
            else:
                sigma = sigma / dps

        dps, dps_err = self.dps_spin_box.value() * 1e-9, self.dps_pm_spin_box.value() * 1e-9
        if dps_err != 0:
            data = signal.get_investigation_data(sigma, dps, dps_err=dps_err, fit_type=fit_type)
        else:
            data = signal.get_investigation_data(sigma, dps, fit_type=fit_type)
        self.data_coherence_length_spin_box.setValue(data['coherence_length'] * 1e6)
        self.data_spectral_width_thz_spin_box.setValue(data['spectral_width_hz'] * 1e-12)
        self.data_spectral_width_nm_spin_box.setValue(data['spectral_width_m'] * 1e9)
        self.data_mean_wavelength_spin_box.setValue(data['mean_wavelength'] * 1e9)
        self.data_mean_frequency_spin_box.setValue(data['mean_frequency'] * 1e-12)

    def update_patches(self, signal):
        if self.use_dist_as_x_checkbox.isChecked():
            dps = self.dps_spin_box.value() * 1e-9
        else:
            dps = None

        signal_plotter.update_patches(signal, self.plot_widget.canvas.ax, dps=dps)
        self.plot_widget.canvas.draw()

    def recalculate(self):
        _signal_name = self.signals_list_widget.currentItem().text()
        _signal_index = self.signals_list_widget.currentRow()
        # print(_signal_name, self.signals_list_widget.currentRow())

        signal = self.data_signals[_signal_index]

        # use_moving_y_offset = self.y_offset_moving_radio_button.isChecked()
        # if use_moving_y_offset:
        #     moving_y_offset_wavelengths = self.y_offset_wavelengths_spin_box.value()
        moving_y_offset_wavelengths = self.y_offset_wavelengths_spin_box.value() if self.y_offset_moving_radio_button.isChecked() else None

        if self.use_abs_checkbox.isChecked():
            signal.preprocess_xy(use_abs=True)
        else:
            signal.preprocess_xy(moving_y_offset_wavelengths=moving_y_offset_wavelengths)
        self.draw_signal(signal)

        # COPY SIGNAL DATA
        cb = QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        # print('hi')
        text = signal.as_string(*signal.get_local_maxes())
        # print(text)
        cb.setText(text, mode=cb.Clipboard)

    def toggle_y_offset_type(self):
        if self.y_offset_moving_radio_button.isChecked():
            self.y_offset_double_spin_box.setDisabled(True)
            self.calculated_y_offset_label.setDisabled(True)
            self.y_offset_wavelengths_label.setDisabled(False)
            self.y_offset_wavelengths_spin_box.setDisabled(False)
        else:
            self.y_offset_wavelengths_label.setDisabled(True)
            self.y_offset_wavelengths_spin_box.setDisabled(True)
            self.y_offset_double_spin_box.setDisabled(False)
            self.calculated_y_offset_label.setDisabled(False)

        self.recalculate()

    def toggle_fit_type(self):
        if self.sender().isChecked():
            if self.exponential_radio_button.isChecked():
                self.fit_s_label.setText(u'\u03b3')
            elif self.lorentzian_radio_button.isChecked():
                self.fit_s_label.setText(u'\u03b3')
            elif self.gaussian_radio_button.isChecked():
                self.fit_s_label.setText(u'\u03c3')

            self.recalculate()

    def toggle_use_beating_freq(self):
        if self.fit_beat_check_box.isChecked():
            self.fit_beat_freq_label.setDisabled(False)
            self.fit_beat_freq_double_spin_box.setDisabled(False)
        else:
            self.fit_beat_freq_label.setDisabled(True)
            self.fit_beat_freq_double_spin_box.setDisabled(True)

        self.recalculate()

    def find_motor_dps_calibration(self):
        _signal_name = self.signals_list_widget.currentItem().text()
        _signal_index = self.signals_list_widget.currentRow()
        # print(_signal_name, self.signals_list_widget.currentRow())

        signal = self.data_signals[_signal_index]

        known_wavelength = self.wavelength_double_spin_box.value() * 1e-9
        signal_plotter.plot_motor_step_dps_with_bins(signal, known_wavelength=known_wavelength)
        signal_plotter.plot_motor_step_dps_per_peak(signal, known_wavelength=known_wavelength)
        signal_plotter.plot_motor_step_dps_with_fourier(signal, known_wavelength=known_wavelength)

    def toggle_x_dist(self):
        use_dist_as_x = self.use_dist_as_x_checkbox.isChecked()

        _signal_name = self.signals_list_widget.currentItem().text()
        _signal_index = self.signals_list_widget.currentRow()
        # print(_signal_name, self.signals_list_widget.currentRow())

        signal = self.data_signals[_signal_index]

        if use_dist_as_x:
            # enable stuff
            self.dps_label.setDisabled(False)
            self.dps_spin_box.setDisabled(False)
        else:
            self.dps_label.setDisabled(True)
            self.dps_spin_box.setDisabled(True)

        self.recalculate()

    def export_to_clipboard(self):
        _signal_name = self.signals_list_widget.currentItem().text()
        _signal_index = self.signals_list_widget.currentRow()
        # print(_signal_name, self.signals_list_widget.currentRow())

        signal = self.data_signals[_signal_index]
        if self.sender() is self.export_full_push_button:
            text = signal.as_string(signal.x, signal.y)
        elif self.sender() is self.export_maxes_push_button:
            text = signal.as_string(*signal.get_local_maxes())
        elif self.sender() is self.export_steps_push_button:
            text = signal.as_string(*signal.get_steps_between_peaks()[1:], fmt=('%i', '%i'))
        elif self.sender() is self.export_ft_push_button:
            _, _, frequencies, magnitudes = signal.get_motor_step_dps_with_fourier(known_wavelength=2)
            text = signal.as_string(frequencies, magnitudes, fmt=('%.18e', '%.18e'))
        cb = QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(text, mode=cb.Clipboard)
