import numpy as np
import scipy.interpolate as si

class Posterior(object):
    def __init__(self, times, intensities, nspline):
        self._times = times
        self._intensities = intensities
        self._nspline = nspline

    @property
    def times(self):
        return self._times

    @property
    def intensities(self):
        return self._intensities

    @property
    def nspline(self):
        return self._nspline

    @property
    def dtype(self):
        nspline = self.nspline
        return np.dtype([('P', np.float),
                         ('T', np.float),
                         ('d', np.float),
                         ('t0', np.float),
                         ('sigma', np.float),
                         ('tau', np.float),
                         ('times', np.float, nspline-2),
                         ('values', np.float, nspline)])

    @property
    def pnames(self):
        fixed_names = [r'$P$', r'$T_\mathrm{dur}$', r'$d$', r'$T_0$', r'$\sigma$',
                       r'$\tau$']
        time_names = ['$t_{{{0:d}}}$'.format(i) for i in range(1, self.nspline-1)]
        value_names = ['$I_{{{0:d}}}$'.format(i) for i in range(self.nspline)]

        return fixed_names + time_names + value_names

    def to_params(self, p):
        p = np.atleast_1d(p)
        return np.squeeze(p.view(self.dtype))

    def transit_depths(self, p):
        p = self.to_params(p)

        dt = self.times[1] - self.times[0]

        depths = np.zeros(self.times.shape)

        left_bounds = self.times - dt/2.0
        right_bounds = self.times + dt/2.0

        left_time_since_transit = np.fmod(left_bounds - p['t0'], p['P'])
        right_time_since_transit = np.fmod(right_bounds - p['t0'], p['P'])

        left_in_transit = (left_time_since_transit > 0) & (left_time_since_transit < p['T'])
        right_in_transit = (right_time_since_transit > 0) & (right_time_since_transit < p['T'])

        depths[left_in_transit & right_in_transit] = 1.0

        entries = right_in_transit & (~left_in_transit)
        exits = left_in_transit & (~right_in_transit)

        depths[entries] = right_time_since_transit[entries] / dt
        depths[exits] = (p['T']-left_time_since_transit[exits])/dt        

        return depths

    def log_prior(self, p):
        p = self.to_params(p)

        N = self.times.shape[0]

        t0 = self.times[1]
        tmax = self.times[-1]

        dT = tmax - t0

        dt = np.mean(np.diff(self.times))

        Imax = np.max(self.intensities)
        dI = Imax - np.min(self.intensities)

        if p['P'] <= 0 or p['P'] >= dT/2.0:
            return np.NINF
        if p['T'] <= 0 or p['T'] >= p['P']:
            return np.NINF
        if p['d'] <= 0.0 or p['d'] > 1.0:
            return np.NINF
        if p['t0'] <= t0 or p['t0'] > t0 + p['P']:
            return np.NINF
        if p['sigma'] <= 0 or p['sigma'] > 10.0*dI:
            return np.NINF
        if p['tau'] < dt/10.0 or p['tau'] > dT*10.0:
            return np.NINF
        if np.any(p['times'] <= t0) or np.any(p['times'] >= tmax) or np.any(p['times'][1:] < p['times'][:-1]):
            return np.NINF
        if np.any(p['values']) <= 0 or np.any(p['values'] > Imax + dI):
            return np.NINF

        return 0.0

    def log_likelihood(self, p):
        p = self.to_params(p)

        N = self.times.shape[0]

        dd_data = self.detransited_detrended_data(p)

        alphas = np.exp(-(self.times[1:] - self.times[:-1])/p['tau'])
        betas = p['sigma']*np.sqrt(1-alphas*alphas)

        decorr_dd_data = dd_data[1:] - alphas*dd_data[:-1]

        return np.sum(-0.5*decorr_dd_data*decorr_dd_data/(betas*betas)) - 0.5*dd_data[0]*dd_data[0]/(p['sigma']*p['sigma']) - 0.5*N*np.log(2.0*np.pi) - np.sum(np.log(betas)) - np.log(p['sigma'])

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return lp
        else:
            return lp + self.log_likelihood(p)

    def data_sample(self, p):
        p = self.to_params(p)

        alphas = np.exp(-np.diff(self.times)/p['tau'])
        betas = p['sigma']*np.sqrt(1-alphas*alphas)

        n = [np.random.randn()*p['sigma']]
        for alpha, beta in zip(alphas, betas):
            n.append(alpha*n[-1] + beta*np.random.randn())
        n = np.array(n)

        sp_times = np.concatenate(([self.times[0]], p['times'], [self.times[-1]]))
        trended_noise = n + si.UnivariateSpline(sp_times, p['values'])(self.times)

        depths = self.transit_depths(p)
        data = trended_noise*(1 - depths*p['d'])

        return data

    def detransited_data(self, p):
        p = self.to_params(p)

        data = self.intensities.copy()

        depths = self.transit_depths(p)

        data = self.intensities / (1 - depths*p['d'])
        
        return data

    def detransited_detrended_data(self, p):
        p = self.to_params(p)

        data = self.detransited_data(p)

        sp_times = np.concatenate(([self.times[0]], p['times'], [self.times[-1]]))
        return data - si.UnivariateSpline(sp_times, p['values'])(self.times)

    def detransited_detrended_decorrelated_data(self, p):
        p = self.to_params(p)

        data = self.detransited_detrended_data(p)

        alphas = np.exp(-(self.times[1:] - self.times[:-1])/p['tau'])
        betas = p['sigma']*np.sqrt(1-alphas*alphas)

        ddd_data = data.copy()
        ddd_data[1:] = ddd_data[1:] - alphas*ddd_data[:-1]

        ddd_data[0] /= p['sigma']
        ddd_data[1:] /= betas

        return ddd_data
