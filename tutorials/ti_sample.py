# This file can be used to run thermodynamic integration
# To run this, use `mpirun -np [NUMBER OF CHAINS] python ti_sample.py`
# Note that this will require a lot of memory!

import numpy as np
import json
import glob
from h5pulsar.pulsar import FilePulsar

from enterprise_extensions import sampler, models

def sample_irn(psrs, noise_params, gwbfreqs=14, Tmax=1e7, num_samples=5e6):
    pta = models.model_1(psrs, noisedict=noise_params, tm_marg=True, tm_svd=True)
    outdir = './chains/15yr_ti_irn_{0}_{1}'.format(gwbfreqs, Tmax)
    emp_dist = datadir + 'emp_distr.pkl'
    sampler1 = sampler.setup_sampler(pta, outdir=outdir, resume=True, empirical_distr=emp_dist)
    N = int(num_samples)
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler1.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=True, Tmax=Tmax, hotChain=True)

def sample_curn(psrs, noise_params, gwbfreqs=14, Tmax=1e7, num_samples=5e6):
    pta = models.model_2a(psrs, noisedict=noise_params, n_gwbfreqs=14, tm_marg=True, tm_svd=True)
    outdir = './chains/15yr_ti_curn_varied_gamma_{0}_{1}'.format(gwbfreqs, Tmax)
    emp_dist = datadir + 'emp_distr.pkl'
    sampler1 = sampler.setup_sampler(pta, outdir=outdir, resume=True, empirical_distr=emp_dist)
    N = int(num_samples)
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler1.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=True, Tmax=Tmax, hotChain=True)

def sample_hd(psrs, noise_params, gwbfreqs=14, Tmax=1e7, num_samples=5e6):
    pta = models.model_3a(psrs, noisedict=noise_params, n_gwbfreqs=14, tm_marg=True, tm_svd=True)
    outdir = './chains/15yr_ti_hd_varied_gamma_{0}_{1}'.format(gwbfreqs, Tmax)
    emp_dist = datadir + 'emp_distr.pkl'
    sampler1 = sampler.setup_sampler(pta, outdir=outdir, resume=True, empirical_distr=emp_dist)
    N = int(num_samples)
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler1.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=True, Tmax=Tmax, hotChain=True)

if __name__ == '__main__':
    datadir = './data/'
    with open(datadir + 'v1p1_wn_dict.json') as f:
        wn_params = json.load(f)

    psrs = []
    for hdf5_file in glob.glob(datadir + '/hdf5/*.hdf5'):
        psrs.append(FilePulsar(hdf5_file))
    print('Loaded {0} pulsars from hdf5 files'.format(len(psrs)))

    # change this to whichever model you would like to sample:
    sample_curn(psrs, wn_params)
