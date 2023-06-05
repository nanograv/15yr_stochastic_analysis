import numpy as np

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
