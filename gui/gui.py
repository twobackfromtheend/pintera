# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(946, 916)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.signals_group_box = QtWidgets.QGroupBox(self.frame)
        self.signals_group_box.setObjectName("signals_group_box")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.signals_group_box)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.signals_list_widget = QtWidgets.QListWidget(self.signals_group_box)
        self.signals_list_widget.setMaximumSize(QtCore.QSize(500, 16777215))
        self.signals_list_widget.setObjectName("signals_list_widget")
        self.horizontalLayout_3.addWidget(self.signals_list_widget)
        self.horizontalLayout_2.addWidget(self.signals_group_box)
        self.v_line_1 = QtWidgets.QFrame(self.frame)
        self.v_line_1.setFrameShape(QtWidgets.QFrame.VLine)
        self.v_line_1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.v_line_1.setObjectName("v_line_1")
        self.horizontalLayout_2.addWidget(self.v_line_1)
        self.signal_options_group_box = QtWidgets.QGroupBox(self.frame)
        self.signal_options_group_box.setObjectName("signal_options_group_box")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.signal_options_group_box)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_6 = QtWidgets.QFrame(self.signal_options_group_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 500))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.signal_options_autosave_check_box = QtWidgets.QCheckBox(self.frame_6)
        self.signal_options_autosave_check_box.setChecked(True)
        self.signal_options_autosave_check_box.setObjectName("signal_options_autosave_check_box")
        self.horizontalLayout_6.addWidget(self.signal_options_autosave_check_box)
        self.pushButton = QtWidgets.QPushButton(self.frame_6)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_6.addWidget(self.pushButton)
        self.signal_options_reset_push_button = QtWidgets.QPushButton(self.frame_6)
        self.signal_options_reset_push_button.setObjectName("signal_options_reset_push_button")
        self.horizontalLayout_6.addWidget(self.signal_options_reset_push_button)
        self.verticalLayout.addWidget(self.frame_6)
        self.line_4 = QtWidgets.QFrame(self.signal_options_group_box)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout.addWidget(self.line_4)
        self.signal_options_x_frame = QtWidgets.QFrame(self.signal_options_group_box)
        self.signal_options_x_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.signal_options_x_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.signal_options_x_frame.setObjectName("signal_options_x_frame")
        self.gridLayout = QtWidgets.QGridLayout(self.signal_options_x_frame)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.signal_options_x_frame)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.signal_options_x_frame)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.signal_options_x_frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.x_lim_spin_box_1 = QtWidgets.QSpinBox(self.frame_2)
        self.x_lim_spin_box_1.setMaximum(999999999)
        self.x_lim_spin_box_1.setObjectName("x_lim_spin_box_1")
        self.horizontalLayout.addWidget(self.x_lim_spin_box_1)
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.x_lim_spin_box_2 = QtWidgets.QSpinBox(self.frame_2)
        self.x_lim_spin_box_2.setMaximum(999999999)
        self.x_lim_spin_box_2.setObjectName("x_lim_spin_box_2")
        self.horizontalLayout.addWidget(self.x_lim_spin_box_2)
        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 3)
        self.gridLayout.addWidget(self.frame_2, 0, 1, 1, 4)
        self.x_centre_spin_box = QtWidgets.QSpinBox(self.signal_options_x_frame)
        self.x_centre_spin_box.setMaximum(999999999)
        self.x_centre_spin_box.setObjectName("x_centre_spin_box")
        self.gridLayout.addWidget(self.x_centre_spin_box, 1, 1, 1, 4)
        self.verticalLayout.addWidget(self.signal_options_x_frame)
        self.line_2 = QtWidgets.QFrame(self.signal_options_group_box)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.recalculate_push_button = QtWidgets.QPushButton(self.signal_options_group_box)
        self.recalculate_push_button.setObjectName("recalculate_push_button")
        self.verticalLayout.addWidget(self.recalculate_push_button)
        self.frame_4 = QtWidgets.QFrame(self.signal_options_group_box)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.y_offset_wavelengths_spin_box = QtWidgets.QSpinBox(self.frame_4)
        self.y_offset_wavelengths_spin_box.setMinimum(1)
        self.y_offset_wavelengths_spin_box.setProperty("value", 5)
        self.y_offset_wavelengths_spin_box.setObjectName("y_offset_wavelengths_spin_box")
        self.gridLayout_4.addWidget(self.y_offset_wavelengths_spin_box, 1, 2, 1, 1)
        self.calculated_y_offset_label = QtWidgets.QLabel(self.frame_4)
        self.calculated_y_offset_label.setEnabled(False)
        self.calculated_y_offset_label.setObjectName("calculated_y_offset_label")
        self.gridLayout_4.addWidget(self.calculated_y_offset_label, 0, 1, 1, 1)
        self.y_offset_double_spin_box = QtWidgets.QDoubleSpinBox(self.frame_4)
        self.y_offset_double_spin_box.setEnabled(False)
        self.y_offset_double_spin_box.setDecimals(5)
        self.y_offset_double_spin_box.setObjectName("y_offset_double_spin_box")
        self.gridLayout_4.addWidget(self.y_offset_double_spin_box, 0, 2, 1, 1)
        self.y_offset_fixed_radio_button = QtWidgets.QRadioButton(self.frame_4)
        self.y_offset_fixed_radio_button.setObjectName("y_offset_fixed_radio_button")
        self.gridLayout_4.addWidget(self.y_offset_fixed_radio_button, 0, 0, 1, 1)
        self.y_offset_moving_radio_button = QtWidgets.QRadioButton(self.frame_4)
        self.y_offset_moving_radio_button.setChecked(True)
        self.y_offset_moving_radio_button.setObjectName("y_offset_moving_radio_button")
        self.gridLayout_4.addWidget(self.y_offset_moving_radio_button, 1, 0, 1, 1)
        self.y_offset_wavelengths_label = QtWidgets.QLabel(self.frame_4)
        self.y_offset_wavelengths_label.setObjectName("y_offset_wavelengths_label")
        self.gridLayout_4.addWidget(self.y_offset_wavelengths_label, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.frame_4)
        self.line_3 = QtWidgets.QFrame(self.signal_options_group_box)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        self.frame_5 = QtWidgets.QFrame(self.signal_options_group_box)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.exponential_radio_button = QtWidgets.QRadioButton(self.frame_5)
        self.exponential_radio_button.setChecked(True)
        self.exponential_radio_button.setObjectName("exponential_radio_button")
        self.horizontalLayout_5.addWidget(self.exponential_radio_button)
        self.lorentzian_radio_button = QtWidgets.QRadioButton(self.frame_5)
        self.lorentzian_radio_button.setChecked(False)
        self.lorentzian_radio_button.setObjectName("lorentzian_radio_button")
        self.horizontalLayout_5.addWidget(self.lorentzian_radio_button)
        self.gaussian_radio_button = QtWidgets.QRadioButton(self.frame_5)
        self.gaussian_radio_button.setObjectName("gaussian_radio_button")
        self.horizontalLayout_5.addWidget(self.gaussian_radio_button)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_5.setStretch(2, 1)
        self.verticalLayout.addWidget(self.frame_5)
        self.frame_7 = QtWidgets.QFrame(self.signal_options_group_box)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.fit_beat_check_box = QtWidgets.QCheckBox(self.frame_7)
        self.fit_beat_check_box.setObjectName("fit_beat_check_box")
        self.horizontalLayout_9.addWidget(self.fit_beat_check_box)
        self.use_abs_checkbox = QtWidgets.QCheckBox(self.frame_7)
        self.use_abs_checkbox.setObjectName("use_abs_checkbox")
        self.horizontalLayout_9.addWidget(self.use_abs_checkbox)
        self.verticalLayout.addWidget(self.frame_7)
        self.line_5 = QtWidgets.QFrame(self.signal_options_group_box)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout.addWidget(self.line_5)
        self.find_dps_push_button = QtWidgets.QPushButton(self.signal_options_group_box)
        self.find_dps_push_button.setObjectName("find_dps_push_button")
        self.verticalLayout.addWidget(self.find_dps_push_button)
        self.frame_3 = QtWidgets.QFrame(self.signal_options_group_box)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.wavelength_double_spin_box = QtWidgets.QDoubleSpinBox(self.frame_3)
        self.wavelength_double_spin_box.setDecimals(2)
        self.wavelength_double_spin_box.setMaximum(9999999.0)
        self.wavelength_double_spin_box.setProperty("value", 546.1)
        self.wavelength_double_spin_box.setObjectName("wavelength_double_spin_box")
        self.horizontalLayout_7.addWidget(self.wavelength_double_spin_box)
        self.verticalLayout.addWidget(self.frame_3)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(2, 3)
        self.verticalLayout.setStretch(4, 3)
        self.verticalLayout.setStretch(5, 3)
        self.horizontalLayout_2.addWidget(self.signal_options_group_box)
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.use_dist_as_x_checkbox = QtWidgets.QCheckBox(self.groupBox_2)
        self.use_dist_as_x_checkbox.setObjectName("use_dist_as_x_checkbox")
        self.verticalLayout_3.addWidget(self.use_dist_as_x_checkbox)
        self.dps_frame = QtWidgets.QFrame(self.groupBox_2)
        self.dps_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dps_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.dps_frame.setObjectName("dps_frame")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.dps_frame)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.dps_label = QtWidgets.QLabel(self.dps_frame)
        self.dps_label.setEnabled(False)
        self.dps_label.setObjectName("dps_label")
        self.horizontalLayout_4.addWidget(self.dps_label)
        self.dps_spin_box = QtWidgets.QDoubleSpinBox(self.dps_frame)
        self.dps_spin_box.setEnabled(False)
        self.dps_spin_box.setDecimals(3)
        self.dps_spin_box.setProperty("value", 2.441)
        self.dps_spin_box.setObjectName("dps_spin_box")
        self.horizontalLayout_4.addWidget(self.dps_spin_box)
        self.dps_pm_spin_box = QtWidgets.QDoubleSpinBox(self.dps_frame)
        self.dps_pm_spin_box.setDecimals(3)
        self.dps_pm_spin_box.setMaximum(999999.0)
        self.dps_pm_spin_box.setObjectName("dps_pm_spin_box")
        self.horizontalLayout_4.addWidget(self.dps_pm_spin_box)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 3)
        self.verticalLayout_3.addWidget(self.dps_frame)
        self.line = QtWidgets.QFrame(self.groupBox_2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_3.addWidget(self.line)
        self.separate_plot_checkbox = QtWidgets.QCheckBox(self.groupBox_2)
        self.separate_plot_checkbox.setObjectName("separate_plot_checkbox")
        self.verticalLayout_3.addWidget(self.separate_plot_checkbox)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 1)
        self.verticalLayout_2.addWidget(self.frame)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.fit_s_label = QtWidgets.QLabel(self.groupBox_4)
        self.fit_s_label.setTextFormat(QtCore.Qt.RichText)
        self.fit_s_label.setObjectName("fit_s_label")
        self.gridLayout_3.addWidget(self.fit_s_label, 2, 0, 1, 1)
        self.fit_s_double_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.fit_s_double_spin_box.setDecimals(9)
        self.fit_s_double_spin_box.setMaximum(999999999.99)
        self.fit_s_double_spin_box.setObjectName("fit_s_double_spin_box")
        self.gridLayout_3.addWidget(self.fit_s_double_spin_box, 2, 1, 1, 1)
        self.fit_m_label = QtWidgets.QLabel(self.groupBox_4)
        self.fit_m_label.setObjectName("fit_m_label")
        self.gridLayout_3.addWidget(self.fit_m_label, 1, 0, 1, 1)
        self.fit_a_double_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.fit_a_double_spin_box.setDecimals(5)
        self.fit_a_double_spin_box.setMaximum(999999999.99)
        self.fit_a_double_spin_box.setObjectName("fit_a_double_spin_box")
        self.gridLayout_3.addWidget(self.fit_a_double_spin_box, 0, 1, 1, 1)
        self.fit_m_double_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.fit_m_double_spin_box.setDecimals(9)
        self.fit_m_double_spin_box.setMaximum(999999999.99)
        self.fit_m_double_spin_box.setObjectName("fit_m_double_spin_box")
        self.gridLayout_3.addWidget(self.fit_m_double_spin_box, 1, 1, 1, 1)
        self.fit_a_label = QtWidgets.QLabel(self.groupBox_4)
        self.fit_a_label.setObjectName("fit_a_label")
        self.gridLayout_3.addWidget(self.fit_a_label, 0, 0, 1, 1)
        self.fit_beat_freq_label = QtWidgets.QLabel(self.groupBox_4)
        self.fit_beat_freq_label.setObjectName("fit_beat_freq_label")
        self.gridLayout_3.addWidget(self.fit_beat_freq_label, 3, 0, 1, 1)
        self.fit_beat_freq_double_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.fit_beat_freq_double_spin_box.setDecimals(9)
        self.fit_beat_freq_double_spin_box.setMaximum(999999999.0)
        self.fit_beat_freq_double_spin_box.setObjectName("fit_beat_freq_double_spin_box")
        self.gridLayout_3.addWidget(self.fit_beat_freq_double_spin_box, 3, 1, 1, 1)
        self.horizontalLayout_8.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.data_mean_wavelength_pm_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_mean_wavelength_pm_spin_box.setDecimals(5)
        self.data_mean_wavelength_pm_spin_box.setMaximum(99999999999.0)
        self.data_mean_wavelength_pm_spin_box.setObjectName("data_mean_wavelength_pm_spin_box")
        self.gridLayout_2.addWidget(self.data_mean_wavelength_pm_spin_box, 6, 2, 1, 1)
        self.data_spectral_width_nm_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_spectral_width_nm_spin_box.setDecimals(5)
        self.data_spectral_width_nm_spin_box.setMaximum(99999999999.0)
        self.data_spectral_width_nm_spin_box.setObjectName("data_spectral_width_nm_spin_box")
        self.gridLayout_2.addWidget(self.data_spectral_width_nm_spin_box, 3, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 2, 0, 2, 1)
        self.data_spectral_width_thz_pm_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_spectral_width_thz_pm_spin_box.setDecimals(5)
        self.data_spectral_width_thz_pm_spin_box.setMaximum(99999999999.0)
        self.data_spectral_width_thz_pm_spin_box.setObjectName("data_spectral_width_thz_pm_spin_box")
        self.gridLayout_2.addWidget(self.data_spectral_width_thz_pm_spin_box, 2, 2, 1, 1)
        self.data_spectral_width_thz_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_spectral_width_thz_spin_box.setDecimals(5)
        self.data_spectral_width_thz_spin_box.setMaximum(99999999999.0)
        self.data_spectral_width_thz_spin_box.setObjectName("data_spectral_width_thz_spin_box")
        self.gridLayout_2.addWidget(self.data_spectral_width_thz_spin_box, 2, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_5)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 5, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_5)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 6, 0, 1, 1)
        self.data_mean_wavelength_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_mean_wavelength_spin_box.setDecimals(5)
        self.data_mean_wavelength_spin_box.setMaximum(99999999999.0)
        self.data_mean_wavelength_spin_box.setObjectName("data_mean_wavelength_spin_box")
        self.gridLayout_2.addWidget(self.data_mean_wavelength_spin_box, 6, 1, 1, 1)
        self.data_coherence_length_pm_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_coherence_length_pm_spin_box.setDecimals(5)
        self.data_coherence_length_pm_spin_box.setMaximum(99999999999.0)
        self.data_coherence_length_pm_spin_box.setObjectName("data_coherence_length_pm_spin_box")
        self.gridLayout_2.addWidget(self.data_coherence_length_pm_spin_box, 0, 2, 1, 1)
        self.data_spectral_width_nm_pm_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_spectral_width_nm_pm_spin_box.setDecimals(5)
        self.data_spectral_width_nm_pm_spin_box.setMaximum(99999999999.0)
        self.data_spectral_width_nm_pm_spin_box.setObjectName("data_spectral_width_nm_pm_spin_box")
        self.gridLayout_2.addWidget(self.data_spectral_width_nm_pm_spin_box, 3, 2, 1, 1)
        self.data_mean_frequency_pm_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_mean_frequency_pm_spin_box.setDecimals(5)
        self.data_mean_frequency_pm_spin_box.setMaximum(99999999999.0)
        self.data_mean_frequency_pm_spin_box.setObjectName("data_mean_frequency_pm_spin_box")
        self.gridLayout_2.addWidget(self.data_mean_frequency_pm_spin_box, 5, 2, 1, 1)
        self.data_coherence_length_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_coherence_length_spin_box.setDecimals(5)
        self.data_coherence_length_spin_box.setMaximum(99999999999.0)
        self.data_coherence_length_spin_box.setObjectName("data_coherence_length_spin_box")
        self.gridLayout_2.addWidget(self.data_coherence_length_spin_box, 0, 1, 1, 1)
        self.line_6 = QtWidgets.QFrame(self.groupBox_5)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout_2.addWidget(self.line_6, 1, 0, 1, 3)
        self.line_7 = QtWidgets.QFrame(self.groupBox_5)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout_2.addWidget(self.line_7, 4, 0, 1, 3)
        self.data_mean_frequency_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.data_mean_frequency_spin_box.setDecimals(5)
        self.data_mean_frequency_spin_box.setMaximum(99999999999.0)
        self.data_mean_frequency_spin_box.setObjectName("data_mean_frequency_spin_box")
        self.gridLayout_2.addWidget(self.data_mean_frequency_spin_box, 5, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setRowStretch(0, 1)
        self.horizontalLayout_8.addWidget(self.groupBox_5)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_4.addWidget(self.pushButton_2)
        self.horizontalLayout_8.addWidget(self.groupBox_3)
        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 2)
        self.horizontalLayout_8.setStretch(2, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.plot_frame = QtWidgets.QFrame(self.centralwidget)
        self.plot_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plot_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plot_frame.setObjectName("plot_frame")
        self.plot_layout = QtWidgets.QHBoxLayout(self.plot_frame)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setObjectName("plot_layout")
        self.plot_widget = QtWidgets.QWidget(self.plot_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        self.plot_widget.setSizePolicy(sizePolicy)
        self.plot_widget.setMinimumSize(QtCore.QSize(0, 300))
        self.plot_widget.setObjectName("plot_widget")
        self.plot_layout.addWidget(self.plot_widget)
        self.verticalLayout_2.addWidget(self.plot_frame)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 21))
        self.menubar.setObjectName("menubar")
        self.menuLoad_Signals = QtWidgets.QMenu(self.menubar)
        self.menuLoad_Signals.setObjectName("menuLoad_Signals")
        self.menuOptions = QtWidgets.QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        self.menuExport = QtWidgets.QMenu(self.menubar)
        self.menuExport.setObjectName("menuExport")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSignal = QtWidgets.QAction(MainWindow)
        self.actionSignal.setObjectName("actionSignal")
        self.actionSignals_from_Folder = QtWidgets.QAction(MainWindow)
        self.actionSignals_from_Folder.setObjectName("actionSignals_from_Folder")
        self.actionShow_Hide_Editing_Panel = QtWidgets.QAction(MainWindow)
        self.actionShow_Hide_Editing_Panel.setObjectName("actionShow_Hide_Editing_Panel")
        self.actionReset_View_Options = QtWidgets.QAction(MainWindow)
        self.actionReset_View_Options.setObjectName("actionReset_View_Options")
        self.actionSignal_CSVs = QtWidgets.QAction(MainWindow)
        self.actionSignal_CSVs.setObjectName("actionSignal_CSVs")
        self.menuLoad_Signals.addAction(self.actionSignal)
        self.menuLoad_Signals.addAction(self.actionSignals_from_Folder)
        self.menuOptions.addAction(self.actionReset_View_Options)
        self.menuExport.addAction(self.actionSignal_CSVs)
        self.menubar.addAction(self.menuLoad_Signals.menuAction())
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Analyser"))
        self.signals_group_box.setTitle(_translate("MainWindow", "Signals"))
        self.signal_options_group_box.setTitle(_translate("MainWindow", "Signal Options"))
        self.signal_options_autosave_check_box.setText(_translate("MainWindow", "Autosave"))
        self.pushButton.setText(_translate("MainWindow", "Save"))
        self.signal_options_reset_push_button.setText(_translate("MainWindow", "Reset"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-style:italic;\">x</span> limits:</p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-style:italic;\">x</span> centre:</p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "to"))
        self.recalculate_push_button.setText(_translate("MainWindow", "Recalculate Fit and y offset"))
        self.calculated_y_offset_label.setText(_translate("MainWindow", "<html><head/><body><p>Calculated <span style=\" font-style:italic;\">y</span> offset:</p></body></html>"))
        self.y_offset_fixed_radio_button.setText(_translate("MainWindow", "Fixed"))
        self.y_offset_moving_radio_button.setToolTip(_translate("MainWindow", "Moving average across n wavelengths"))
        self.y_offset_moving_radio_button.setText(_translate("MainWindow", "Moving"))
        self.y_offset_wavelengths_label.setToolTip(_translate("MainWindow", "Number of wavelengths to calculate y offset across"))
        self.y_offset_wavelengths_label.setText(_translate("MainWindow", "Wavelengths:"))
        self.exponential_radio_button.setToolTip(_translate("MainWindow", "A * exp(-γ * abs(x - m))"))
        self.exponential_radio_button.setText(_translate("MainWindow", "Exponential"))
        self.lorentzian_radio_button.setToolTip(_translate("MainWindow", "A / (1 + ((x - m) / γ)^2)"))
        self.lorentzian_radio_button.setText(_translate("MainWindow", "Lorentzian"))
        self.gaussian_radio_button.setToolTip(_translate("MainWindow", "A * exp(-(x - m)^2 / (2 * σ^2))"))
        self.gaussian_radio_button.setText(_translate("MainWindow", "Gaussian"))
        self.fit_beat_check_box.setToolTip(_translate("MainWindow", " * cos(2πf * x - m)"))
        self.fit_beat_check_box.setText(_translate("MainWindow", "Fit Beating"))
        self.use_abs_checkbox.setToolTip(_translate("MainWindow", "Use absolute values (so negative side of spectra is used too). Not compatible with moving y offset."))
        self.use_abs_checkbox.setText(_translate("MainWindow", "Use absolute values"))
        self.find_dps_push_button.setText(_translate("MainWindow", "Find Motor DPS (Calibration)"))
        self.label_4.setText(_translate("MainWindow", "Known Wavelength:"))
        self.wavelength_double_spin_box.setSuffix(_translate("MainWindow", " nm"))
        self.groupBox_2.setTitle(_translate("MainWindow", "View Options"))
        self.use_dist_as_x_checkbox.setText(_translate("MainWindow", "Use Distance as x"))
        self.dps_label.setToolTip(_translate("MainWindow", "Displacement per Motor Step"))
        self.dps_label.setText(_translate("MainWindow", "DPS:"))
        self.dps_spin_box.setSuffix(_translate("MainWindow", " nm"))
        self.dps_pm_spin_box.setPrefix(_translate("MainWindow", "± "))
        self.dps_pm_spin_box.setSuffix(_translate("MainWindow", " nm"))
        self.separate_plot_checkbox.setText(_translate("MainWindow", "Use separate plot window"))
        self.groupBox.setTitle(_translate("MainWindow", "Signal Data"))
        self.groupBox_4.setTitle(_translate("MainWindow", "SciPy Gaussian Fit"))
        self.fit_s_label.setText(_translate("MainWindow", "γ"))
        self.fit_m_label.setText(_translate("MainWindow", "Mean"))
        self.fit_a_label.setText(_translate("MainWindow", "Amplitude"))
        self.fit_beat_freq_label.setText(_translate("MainWindow", "Beating Frequency"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Data"))
        self.data_mean_wavelength_pm_spin_box.setPrefix(_translate("MainWindow", "± "))
        self.data_spectral_width_nm_spin_box.setSuffix(_translate("MainWindow", " nm"))
        self.label_3.setToolTip(_translate("MainWindow", "c * coherence length / lambda^2"))
        self.label_3.setText(_translate("MainWindow", "Spectral width"))
        self.data_spectral_width_thz_pm_spin_box.setPrefix(_translate("MainWindow", "± "))
        self.data_spectral_width_thz_spin_box.setSuffix(_translate("MainWindow", " THz"))
        self.label_11.setText(_translate("MainWindow", "Mean Frequency"))
        self.label_2.setToolTip(_translate("MainWindow", "FWHM (2 sqrt(2 ln 2) * sigma)"))
        self.label_2.setText(_translate("MainWindow", "Coherence length"))
        self.label_7.setText(_translate("MainWindow", "Mean Wavelength"))
        self.data_mean_wavelength_spin_box.setSuffix(_translate("MainWindow", " nm"))
        self.data_coherence_length_pm_spin_box.setPrefix(_translate("MainWindow", "± "))
        self.data_spectral_width_nm_pm_spin_box.setPrefix(_translate("MainWindow", "± "))
        self.data_mean_frequency_pm_spin_box.setPrefix(_translate("MainWindow", "± "))
        self.data_coherence_length_spin_box.setSuffix(_translate("MainWindow", " μm"))
        self.data_mean_frequency_spin_box.setSuffix(_translate("MainWindow", " THz"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Export"))
        self.pushButton_2.setText(_translate("MainWindow", "Maxes"))
        self.menuLoad_Signals.setTitle(_translate("MainWindow", "Load..."))
        self.menuOptions.setTitle(_translate("MainWindow", "Options"))
        self.menuExport.setTitle(_translate("MainWindow", "Export..."))
        self.actionSignal.setText(_translate("MainWindow", "Signal"))
        self.actionSignals_from_Folder.setText(_translate("MainWindow", "Signals from Folder"))
        self.actionShow_Hide_Editing_Panel.setText(_translate("MainWindow", "Show/Hide Options Panel"))
        self.actionReset_View_Options.setText(_translate("MainWindow", "Reset View Options"))
        self.actionSignal_CSVs.setText(_translate("MainWindow", "Signal CSVs"))

