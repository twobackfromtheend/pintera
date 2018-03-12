# pintera

The Python INTERferogram Analyser (name suggestions accepted).

## Features

### Data Parsing
- Parses data from the files saved from LabView, and allows for easy saving in a .csv format through drag-and-drop.
- Drag and drop files saved from LabView onto data_io.py. The corresponding .csv files will be saved alongside their original data files (e.g. dragging and dropping 'W1S2' will create 'W1S2.csv' in the same folder). You can drag and drop multiple files to parse them all at once.

## Analysis
### Setup
1. Replace the files in data/ with your own files (that were saved straight from LabView).
2. Run start.py to pull up the GUI.

### Signal analysis
Valid data files in the data/ folder are parsed and listed. Select one to view it.
 
 #### Signal Preprocessing
 
 - *x* limits
	- *x* limits (in motor steps) indicate the region to be analysed. This is to filter out bad data on either side of the signal to be analysed. These limits are shown in the preview graph as light gray regions cover the areas beyond the limits indicated. The light gray regions in the graph are automatically updated when the values in the *x* limit boxes are changed.
	- *x* centre is not yet implemented.
- *y* offset
	- Fixed *y* offset calculates the mean *y* over the region within the *x* limits indicated. The entire *y* is shifted down by that amount, so the graph has an average *y* value of 0.
	- Moving *y* offset calculates the mean *y* of the ±*n* wavelengths for each point within the region within the *x* limits indicated. NB. A "wavelength" is calculated as a simple "average distance between peaks" and may not be entirely accurate. However, it serves as a good measure of the range over which each point is averaged.
	- The moving *y* offset is good for correcting for significant shifts in *y* over the course of a signal, and renders more of the signal to be available for use. However, it does create edge effects (where if a good region follows directly after a region that was too high, the starting part of the good region will be likewise shifted down). These edge effects are limited to the number of "wavelengths" indicated, and should be negligible if the offset regions are large.
- The Signal Plot
	- "Adjusted Intensity" refers to the intensity of light measured by the photodiode, shifted down by a fixed *y* offset such that the entire signal has a mean of 0.
	- The *x* axis can be in terms of motor steps (as in the raw data), or if "Use Distance as x" is ticked, in terms of displacement (using the DPS given in View Options).
	- The "Full" plot is of the original full plot, simply shifted down by a constant amount. The data points are shown with tiny 'x's.
	- "Preprocessed" refers to the data after it has been filtered (by the *x* limits), and shifted (by either a Fixed or Moving *y* offset).
	- The "Maxima" plotted are of the "Preprocessed" data.
	- The SciPy fit is a Gaussian fit to the "Maxima" using SciPy.optimise.

#### Other Analysis

##### Calibration
- "Find Motor DPS" calculates the displacement from each motor step (DPS being displacement per step).
- Two methods are used (and plotted and printed):
	1. 
		- The signal is split into *n* equal parts. The DPS is calculated from each of those parts. NumPy.mean and .std are used to estimate the mean and standard deviation from the DPSes calculated from the *n* bins. *n* is plotted from 1 (with no error) to some fraction of the total number of something I forgot. (Can find from code.) To obtain a final value and error, the average mean and average standard deviation for 5 ≤ *n* ≤ 10 is taken (and plotted in red. The horizontal red error bar is not an error, and is simply used to indicate the region over which the mean and standard deviation was averaged). 
		- The result is printed in higher precision than shown on the graph as "Average DPS for 5-10 bins".
	3.  
		- The number of motor steps between each maxima and the next is counted. (This number of steps is a multiple of the "steps per sample" (delta) from the data collection window.) This number of steps, and how many times that number of steps happened between maxima is plotted in the first plot in Figure 2. The mean and standard deviation is calculated using NumPy, and the Gaussian fit is used to show the calculated mean and standard deviation
		- For each "number of motor steps between maxima", the corresponding DPS is calculated using the given *known wavelength*. This is plotted in the second plot in Figure 2. The mean and standard deviation is calculated using NumPy, and the Gaussian fit is used to show the calculated mean and standard deviation. Sidenote: SciPy.optimise is used to find the amplitude of the Gaussian fit, but that has no impact on results, as the Gaussian curve is only used to give a feel for the calculated mean and standard deviation.
		- The result is printed in higher precision than shown on the graphs as "DPS: #, DPS std dev: #".

##### Coherence Length and Spectral Width
NB. The DPS (and associated error) indicated in View Options is important and is used for these calculations.

- A Gaussian is fitted to the maxima using SciPy.optimise. This fitted Gaussian is used to calculate the coherence length (FWHM = 2 sqrt (2 ln 2) σ).
	- The spectral width is calculated (in Hz and m) as 
		- spectral_width_hz = c / (π * coherence_length) [Hz]
		- spectral_width_m = spectral_width_hz * mean_wavelength^2 / c [m]


## TODO
Still adding analysis features.

Right now, clicking "Recalculate" copies the maxes to clipboard (can be pasted in Origin/Excel. I'll move it to the "Export" section someday. Some spinboxes still need to be fixed. Till then, the values are printed and can be read off the console.

I'm looking into using a Fourier Transform method for finding mean wavelength, and have managed to get a reasonable Fourier Transform (can be plotted in terms of magnitude against frequency). Still have to Gaussian fit that and interpret the frequency in terms of distance, then convert that to actual light frequencies.