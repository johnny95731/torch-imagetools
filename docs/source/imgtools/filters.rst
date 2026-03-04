:mod:`imgtools.filters`
=======================

.. currentmodule:: imgtools.filters

Basic spatial and frequency domain filters.

- Spatial domain filters
    * Edge: Laplacian, Sobel, Kirsch, etc.
    * Blurring filters: box blur, Gaussian, guided filter
- Frequency domain filters: based on rfft2
    * Laplacian
    * Gaussian highpass/lowpass
    * Butterworthhighpass/lowpass.

---------


Links
-----

======
_edges
======

.. autosummary::
   :nosignatures:

   gradient_magnitude
   kirsch
   laplacian
   prewitt
   robinson
   scharr
   sobel


====
Blur
====

.. autosummary::
   :nosignatures:

   box_blur
   gaussian_blur
   get_gaussian_kernel
   guided_filter


====
Rfft
====

.. autosummary::
   :nosignatures:

   get_butterworth_highpass
   get_butterworth_lowpass
   get_freq_laplacian
   get_gaussian_highpass
   get_gaussian_lowpass

---------


Documents
---------

.. automodule:: imgtools.filters
   :members:
   :no-docstring:
   :member-order: bysource
