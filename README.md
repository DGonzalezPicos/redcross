# redcross: Reduction and Cross-correlation of High-Resolution spectroscopy for exoplanet atmospheres

This repository contains a set of utilities to work with high-resolution order-by-order spectra, in particular
- Read and visualise time-series spectroscopy 
- Define a reduction routine (normalise, remove continuum, SysRem, masking, ...)
- Cross-correlation: Compute the CCF function for a given atmospheric template
- Visualise the results in different reference frames and generate Kp-V_sys map

This package is intended to work with any instrument provided a datacube with the following data:
 - Wave vector (for each order)
 - Flux matrix (for each order and each frame)
 - Flux error matrix (optional, same shape as Flux)
 
 The main object `Datacube` can be directly manipulated using attribute functions e.g. datacube.function(), which already applies the changes to the current
 datacube. By passing an `plt.axes()` object it automatically shows the new datacube. 

