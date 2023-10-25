import numpy as np
from emcee.autocorr import integrated_time
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson

rng = np.random.default_rng()  # instatiate the RNG

class EmpiricalDistribution2D(object):
    def __init__(self, ed2d):
        self.ndim = ed2d['ndim']
        self.param_names = ed2d['param_names']
        self._Nbins = np.array(ed2d['_Nbins'])

        self._edges = np.array(ed2d['_edges'])
        self._wids = np.array(ed2d['_wids'])

        self._pdf = np.array(ed2d['_pdf'])
        self._cdf = np.array(ed2d['_cdf'])

        self._logpdf = np.array(ed2d['_logpdf'])

    def draw(self):
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw)
        idx = np.unravel_index(draw_bin, self._Nbins)
        samp = [self._edges[ii, idx[ii]] + self._wids[ii, idx[ii]]*np.random.rand()
                for ii in range(2)]
        return np.array(samp)

    def prob(self, params):
        ix, iy = [np.searchsorted(self._edges[ii], params[ii]) - 1 for ii in range(2)]

        return self._pdf[ix, iy]

    def logprob(self, params):
        ix, iy = [np.searchsorted(self._edges[ii], params[ii]) - 1 for ii in range(2)]

        return self._logpdf[ix, iy]

def bootstrap(core, param, num_reals=4000):
    """
    Bootstrap samples (with replacement) a 1d array of nearly independent samples,
    giving a representative subsample for each realization. By taking a mean and
    standard deviation of the resulting subsamples, we can get an uncertainty
    estimate.

    Note on number of samples: The number of samples is set at 1000,
    but in general we should plot the histograms and make sure that they
    look like distributions and not random points.

    Input:
        core (*Core): Any core object
        param (str): parameter to bootstrap sample
        num_reals (int) [2000]: number of realizations
    Output:

    """
    tau = int(np.ceil(integrated_time(core.get_param(param))))
    array = core.get_param(param, thin_by=tau)
    new_array = rng.choice(array, (array.size, num_reals))
    return new_array

def ti_log_evidence(slices_core, verbose=True, bs_iterations=2000,
                    num_samples=1000, plot=False):
    """
    Compute ln(evidence) of chains of several different temperatures.

    Input:
        core (Core): Core containing pt_chains
        verbose (bool) [True]: get more info
        bs_iterations (int) [2000]: number of iterations to use to get error estimate
        num_samples (int) [1000]: number of samples to get from chain (w/ replacement)

    Return:
        ln_Z (float): natural logarithm of the evidence
        total_unc (float): uncertainty in the natural logarithm of the evidence
    """
    # sort params by temperature
    temps = np.array(sorted([float(param) for param in slices_core.params]))
    inv_temps = 1 / temps[::-1]
    num_chains = len(inv_temps)
    # bootstrap:
    new_means = np.zeros((bs_iterations, num_chains))
    for ii in range(num_chains):
        bs = bootstrap(slices_core, str(temps[ii]), num_reals=bs_iterations)
        new_means[:, ii] = np.mean(bs, axis=0)
    new_means = np.flip(new_means)  # we flipped inv_temps, so this should be too!
    # the following line doesn't guarantee monotonicity, but will help get closer to it...
    new_means = np.sort(new_means, axis=1)  # sort because the function should realy be monotonic

    if plot:
        plt.figure(figsize=(12, 5))
        for ii in range(bs_iterations):
            plt.semilogx(inv_temps, new_means[ii, :], color='blue', alpha=0.01)
        plt.semilogx(inv_temps, np.mean(new_means, axis=0), color='red', alpha=1)
        for ii in range(len(inv_temps)):
            plt.axvline(inv_temps[ii], color='k', linestyle='--')
        plt.xlim([1e-10, 1])
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$\left<{\beta\,\mathrm{lnlikelihood}}\right>$')
        plt.show()
        plt.clf()

    ln_Z_arr = np.zeros(bs_iterations)

    x = np.log10(inv_temps)  # interpolate on a log(inv_temp) scale
    x_new = np.linspace(x[0], x[-1], num=10000)  # new interpolated points
    for ii in range(bs_iterations):
        y = new_means[ii, :]
        y_spl = interp1d(x, y)
        ln_Z = simpson(y_spl(x_new), 10**(x_new))
        ln_Z_arr[ii] = ln_Z
    ln_Z = np.mean(ln_Z_arr)
    total_unc = np.std(ln_Z_arr)

    if verbose:
        print()
        print('model:')
        print('$ln(evidence) =', ln_Z)
        print('error in ln_Z =', total_unc)
        print()
    if plot:
        return ln_Z, total_unc, inv_temps, np.mean(new_means, axis=0)
    else:
        return ln_Z, total_unc

