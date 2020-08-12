import numpy as np

from probgf.discretization import Discretization
from probgf.validation import HIDE_VAL


class SpatioTemporalBase:


    @classmethod
    def method_id(cls):
        if 'Base' in cls.__name__: # only base class that should not be used
            return ''
        return cls.__name__
    

    def __init__(self, config, dates, console, emiters):
        self.dates = dates
        self.cons = console
        self.emiters = emiters
        self.per_iter = 100.0 / emiters
        self.discret = None
        self.slice_shape = None
        self.configure(config)


    def name(self):
        raise NotImplementedError


    def configure(self, config):
        self.config = config


    def run_training(self, data, obs, split, progr_train):
        progr_train[split] = 100.0
        self.cons.progress(progr_train, split)


    def run_prediction(self, to_pred, obs, split, progr_pred):
        progr_pred[split] = 100.0
        self.cons.progress(progr_pred, split)
        return np.zeros_like(to_pred)


    def discretize(self, data, obs, split):
        data = data[obs]
        data_disc = np.full(obs.shape[:-1], fill_value=HIDE_VAL, dtype=np.uint16)
        d_labels = self.discret.discretize(split, data.reshape((data.size // obs.shape[3], obs.shape[3])))
        obs = obs[:, :, :, 0]
        data_disc[obs] = d_labels
        return data_disc, obs


class TemporalBase(SpatioTemporalBase):
    """
    processes the spatio-temporal data by processing each single univariate time series
    for each time series process_series is called, which needs to be implemented
    """


    def run_prediction(self, to_pred, obs, split, progr_pred):
        to_pred = np.moveaxis(to_pred, 1, 0) # format from (n, T, V, D) to (T, n, V, D)
        obs = np.moveaxis(obs, 1, 0)
        shape = to_pred.shape
        to_pred = to_pred.reshape(shape[0], np.prod(shape[1:]))
        obs = obs.reshape(shape[0], np.prod(shape[1:]))
        for row in range(to_pred.shape[1]):
            if row % 1000 == 0: # do not write in every iteration
                progr_pred[split] = float(row) / to_pred.shape[1] * 100
                self.cons.progress(progr_pred, split)
            self.process_series(to_pred[:, row], obs[:, row])
        super().run_prediction(to_pred, obs, split, progr_pred) # for final console output
        to_pred = np.moveaxis(to_pred.reshape(shape), 0, 1) # reset format
        return to_pred


    def process_series(self, series, observed, full_replace=False):
        raise NotImplementedError


class LIN(TemporalBase):
    """
    Temporally interpolates the masked values (no further configuration possible)
    Implementation based on scipy.interpolate.interp1d
    """


    @classmethod
    def default_config(cls):
        return ''
    

    def __init__(self, config, dates, console, emiters):
        try:
            from scipy import interpolate
        except Exception:
            raise RuntimeError('Import error, please make sure that "SciPy" is correctly installed for using {}!'.format(self.__class__.method_id()))
        self.interp = interpolate.interp1d
        super().__init__(config, dates, console, emiters)


    def name(self):
        return self.__class__.method_id()


    def configure(self, config):
        self.config = 'linear'


    def process_series(self, series, observed, full_replace=False):
        obs_series = series[observed]
        obs_dates = self.dates[observed]
        if obs_series.size == 0:
            series[:] = 0
        else:
            if obs_series.size == 1: # needs at least two for computing interpolation
                obs_dates = np.array([0, self.dates.max()]).astype(int)
                obs_series = np.array([obs_series[0], obs_series[0]]).astype(int)
            f = self.interp(obs_dates, obs_series, kind=self.config, bounds_error=False, fill_value=(obs_series[0], obs_series[-1]))
            if full_replace:
                series[:] = f(self.dates).astype(series.dtype)
            else:
                series[np.invert(observed)] = f(self.dates[np.invert(observed)]).astype(series.dtype)


class NN(LIN):
    """
    Uses nearest temporally available values (no further configuration possible)
    Implementation based on scipy.interpolate.interp1d
    """


    def configure(self, config):
        self.config = 'nearest'


class HANTS(TemporalBase):
    """
    Implementation of Harmonic ANalysis of Time Series (HANTS) algorithm
    Computes a trigonometric regression, which is used to predict masked values
    Uses least-squares fitting implemented in SciPy
    Adaption: Only fitting once, no iterative re-fitting
    Therefore only the NOF (number of frequencies) is required
    The NOF for each fitted series is adapted
    to the number of available observations
    More info:
    G.J. Roerink, Massimo Menenti, and Wout Verhoef.
    "Reconstructing cloudfree NDVI composites 
    using Fourier analysis of time series."
    (2000) http://doi.org/10.1080/014311600209814
    """


    @classmethod
    def default_config(cls):
        return '2,365'
    
    
    def __init__(self, config, dates, console, emiters):
        super().__init__(config, dates, console, emiters)
        try:
            from scipy.optimize import leastsq
        except Exception:
            raise RuntimeError('Import error, please make sure that "SciPy" is correctly installed for using {}!'.format(self.__class__.method_id()))
        self.lsq = leastsq
        self.theta = np.zeros((self.nof * 2 - 1)) # Fitted HANTS parameters
        self.dates_hants = self.dates / self.days * 2 * np.pi        


    def name(self):
        return '{}_NOF{}_D{}'.format(self.__class__.method_id(), self.nof, self.days)


    def configure(self, config):
        if len(config.split(',')) != 2 \
           or not config.split(',')[0].isdigit() or int(config.split(',')[0]) < 1 \
           or not config.split(',')[1].isdigit() or int(config.split(',')[1]) < 1:
            raise RuntimeError('Invalid config "{}".\n{} needs two comma-seperated integers, first one > 0 denoting the NOF (number of frequencies) and second > 0 denoting the length of the time period!'.format(config, self.__class__.method_id()))
        if int(config.split(',')[1]) <= self.dates.max(): raise RuntimeError('Invalid config "{}".\nLength of time period must be larger than the maximal date in the data ({})!'.format(config, self.dates.max()))
        self.nof, self.days = [int(val) for val in config.split(',')]


    def process_series(self, series, observed, full_replace=False):
        if np.all(np.invert(observed)):
            series[:] = 0
        else:
            usable = min(np.count_nonzero(observed), self.theta.size)
            theta = self.theta[:max(((usable - 1) * 2 - 1), 1)] # harmonic fit should be as complex as possible with number of available observations
            theta = self.lsq(self.compute_HANTS, theta, args=(self.dates_hants[observed], series[observed]))[0]
            if full_replace:
                series[True] = self.compute_HANTS(theta, self.dates_hants[True]).astype(series.dtype)
            else:
                series[np.invert(observed)] = self.compute_HANTS(theta, self.dates_hants[np.invert(observed)]).astype(series.dtype)

    
    @staticmethod
    def compute_HANTS(theta, x, y=None):
        res = theta[-1]
        for idx in range(theta.size // 2):
            res += theta[idx * 2] * np.cos(x * (idx + 1) + theta[idx * 2 + 1])
        if y is None:
            return res
        return res - y
