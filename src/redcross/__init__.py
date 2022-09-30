from .datacube import Datacube
from .cross_correlation import CCF, KpV
from .template import Template
from .planet import Planet
from .read import read_harpsn, read_giano
from .pipeline import Pipeline
from .sysrem import SysRem


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev
import astropy.units as u
import astropy.constants as const
from .datacube import Datacube
from pathos.pools import ProcessPool
__author__ = 'Darío González Picos'
__license__ = 'GPLv3'
__version__ = '0.1'
__email__ = 'dariogonzalezpicos@gmail.com'
__status__ = 'Development'