# This file can be used to run thermodynamic integration
# To run this, use `mpirun -np [NUMBER OF CHAINS] python ti_sample.py`
# Note that this will require a lot of memory!

import numpy as np
import json

import numpy as np
from enterprise_extensions import sampler, models

import pickle

def sample_2a(psrs, noise_params, gwbfreqs=14, Tmax=1e7, num_samples=5e6):
    pta = models.model_2a(psrs, gamma_common=4.33, noisedict=noise_params, n_gwbfreqs=14, tm_marg=True, tm_svd=True)
    outdir = './chains/15yr_ti_m2a_gamma_4.33_{0}_{1}'.format(gwbfreqs, Tmax)
    emp_dist = datadir + 'emp_distr.pkl'
    sampler1 = sampler.setup_sampler(pta, outdir=outdir, resume=True, human='Aaron', empirical_distr=emp_dist)
    N = int(num_samples)
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler1.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=True, Tmax=Tmax, hotChain=True)

def sample_3a(psrs, noise_params, gwbfreqs=14, Tmax=1e7, num_samples=5e6):
    pta = models.model_3a(psrs, noisedict=noise_params, n_gwbfreqs=14, tm_marg=True, tm_svd=True)
    outdir = './chains/15yr_ti_m3a_{0}_{1}'.format(gwbfreqs, Tmax)
    emp_dist = datadir + 'emp_distr.pkl'
    sampler1 = sampler.setup_sampler(pta, outdir=outdir, resume=True, human='Aaron', empirical_distr=emp_dist)
    N = int(num_samples)
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler1.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=True, Tmax=Tmax, hotChain=True)

if __name__ == '__main__':
    datadir = './data/'
    with open(datadir + 'v1p1_wn_dict.json') as f:
        wn_params = json.load(f)
    pickle_loc = datadir + 'v1p1_de440_pint_bipm2019.pkl'
    with open(pickle_loc, 'rb') as f:
        psrs = pickle.load(f)

    sample_2a(psrs, wn_params)
