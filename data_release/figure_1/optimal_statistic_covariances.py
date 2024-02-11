import numpy as np
import scipy.linalg
from enterprise.signals import utils
import math
from tqdm import tqdm
import functools

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def partition(a, n):
    k = len(a) // n
    ps = [a[i:i+k] for i in range(0,len(a),k)]

    return ps

def partition2(a, n):
    js = np.linspace(0, len(a) + 1, n + 1).astype(int)
    return [a[js[i]:js[i + 1]] for i in range(n)]


def weightedavg(rho, sig):
    avg = sum(r / s**2 for r, s in zip(rho, sig))
    weights = sum(1 / s**2 for s in sig)

    return (avg / weights, math.sqrt(1 / weights))


def makebins(os, bins, angles=None):
    angles = os.angles if angles is None else angles

    return np.array([(np.mean(angles[p]), *weightedavg(os.rhos[p], os.sigmas[p]))
                    for p in partition2(np.argsort(angles), bins)])

def makebins2(os, bin_edges, angles=None):
    angles = os.angles if angles is None else angles
    bin_inds = np.digitize(angles, bins)-1
    return np.array([(np.mean(angles[p]), *weightedavg(os.rhos[p], os.sigmas[p]))
                    for p in bin_inds])

def get_HD_curve(zeta):
    
    coszeta = np.cos(zeta)
    xip = (1.-coszeta) / 2.
    HD = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
    
    return HD/2

class Cmat:
    def __init__(self, sm, params):
        self.Nmat = sm.get_ndiag(params)
        self.Sigma = scipy.linalg.cho_factor(sm.get_TNT(params) + np.diag(sm.get_phiinv(params)))
        self.Fmat = sm.get_basis(params)

    def solve(self, r, l):
        FNr = self.Nmat.solve(r, self.Fmat)
        FNl = FNr if (l is r) else self.Nmat.solve(l, self.Fmat)
        return self.Nmat.solve(r, l) - np.dot(FNl.T, scipy.linalg.cho_solve(self.Sigma, FNr))

class OS:
    def __init__(self, psrs, pta, params, residuals=None):
        self.psrs, self.params = psrs, params
        self.pairs  = [(i,j) for i in range(len(pta)) for j in range(i+1, len(pta))]
        self.angles = np.array([np.arccos(np.dot(self.psrs[i].pos, self.psrs[j].pos)) for (i,j) in self.pairs])

        ys = [sm._residuals for sm in pta] if residuals is None else residuals
        self.Cs = [Cmat(sm, params) for sm in pta]

        self.Fgws  = [sm['gw'].get_basis(params) for sm in pta]
        self.Gamma = np.array(pta[0]['gw'].get_phi({**params, 'gw_log10_A': 0})).copy()

        self.phimat = pta[0]['gw'].get_phi({**params, 'gw_log10_A': 0, 'gw_gamma': 0})

        Nfreqs = int(self.Gamma.size / 2)

        FCys = [C.solve(y, Fgw) for C, y, Fgw in zip(self.Cs, ys, self.Fgws)]

        self.FCFs = [C.solve(Fgw, Fgw) for C, Fgw in zip(self.Cs, self.Fgws)]

        # a . np.diag(g) . b = a . (g * b) = (a * g) . b
        ts = [np.dot(FCys[i], self.Gamma * FCys[j]) for (i,j) in self.pairs]
        # A . np.diag(g) . B = (A * g) . B
        self.bs = [np.trace(np.dot(self.FCFs[i] * self.Gamma, self.FCFs[j] * self.Gamma)) for (i,j) in self.pairs]

        self.rhos = np.array(ts) / np.array(self.bs)
        self.sigmas = 1.0 / np.sqrt(self.bs)

        self.rhos_freqs = np.zeros((Nfreqs, self.rhos.size))
        self.sigmas_freqs = np.zeros((Nfreqs, self.rhos.size))

        for ii in range(Nfreqs):
            # pick out just want we want
            gamma_tmp = np.zeros(Nfreqs * 2)
            gamma_tmp[2*ii:2*(ii+1)] = self.phimat[2*ii:2*(ii+1)]
            # a . np.diag(g) . b = a . (g * b) = (a * g) . b
            ts_tmp = [np.dot(FCys[i], gamma_tmp * FCys[j]) for (i,j) in self.pairs]
            # A . np.diag(g) . B = (A * g) . B
            bs_tmp = [np.trace(np.dot(self.FCFs[i] * gamma_tmp, self.FCFs[j] * gamma_tmp)) for (i,j) in self.pairs]

            self.rhos_freqs[ii] = np.array(ts_tmp) / np.array(bs_tmp)
            self.sigmas_freqs[ii] = 1.0 / np.sqrt(bs_tmp)


    def set_residuals(self, residuals):
        ys = residuals

        FCys = [C.solve(y, Fgw) for C, y, Fgw in zip(self.Cs, ys, self.Fgws)]
        ts = [np.dot(FCys[i], self.Gamma * FCys[j]) for (i,j) in self.pairs]
        Nfreqs = int(self.Gamma.size / 2)
        self.rhos = np.array(ts) / np.array(self.bs)

        self.rhos_freqs = np.zeros((Nfreqs, self.rhos.size))
        self.sigmas_freqs = np.zeros((Nfreqs, self.rhos.size))

        for ii in range(Nfreqs):
            # pick out just want we want
            gamma_tmp = np.zeros(Nfreqs * 2)
            # gamma_tmp[2*ii:2*(ii+1)] = self.Gamma[2*ii:2*(ii+1)]
            gamma_tmp[2*ii:2*(ii+1)] = self.phimat[2*ii:2*(ii+1)]
            # a . np.diag(g) . b = a . (g * b) = (a * g) . b
            ts_tmp = [np.dot(FCys[i], gamma_tmp * FCys[j]) for (i,j) in self.pairs]
            # A . np.diag(g) . B = (A * g) . B
            bs_tmp = [np.trace(np.dot(self.FCFs[i] * gamma_tmp, self.FCFs[j] * gamma_tmp)) for (i,j) in self.pairs]

            self.rhos_freqs[ii] = np.array(ts_tmp) / np.array(bs_tmp)
            self.sigmas_freqs[ii] = 1.0 / np.sqrt(bs_tmp)

    def setup_mcos(self, orfs):
        if not hasattr(self, "_mcos_ready") or not self._mcos_ready:
            self.design = np.array([[orfs[kk](self.psrs[i].pos, self.psrs[j].pos) for (i,j) in self.pairs]
                           for kk in range(len(orfs))]).T
            self.sigma_inv_matrix = np.linalg.inv(self.design.T @ np.linalg.inv(np.diag(self.sigmas**2)) @ self.design)
            self.prefac_matrix = self.sigma_inv_matrix @ self.design.T @ np.diag(self.sigmas**-2)
            self._mcos_ready = True


    def mcos(self, orfs=[utils.monopole_orf, utils.dipole_orf, utils.hd_orf]):
        self.setup_mcos(orfs)
        ahat = self.prefac_matrix @ self.rhos
        return ahat, np.sqrt(np.diag(self.sigma_inv_matrix))



    def _set_orf(self, orf):
        if not hasattr(self, '_orf') or self._orf is not orf:
            self.orfs = np.array([orf(self.psrs[i].pos, self.psrs[j].pos) for (i,j) in self.pairs])
            self._orf = orf

    def os(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)
        return (np.sum(self.rhos[mask] * self.orfs[mask] / self.sigmas[mask]**2) /
                np.sum(self.orfs[mask]**2 / self.sigmas[mask]**2))

    def os_frequencies(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)

        return np.sum(self.orfs * self.rhos_freqs * self.sigmas_freqs**-2, axis=1) / np.sum(self.orfs**2 * self.sigmas_freqs**-2, axis=1)

    def os_frequencies_sigmas(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)

        return np.sum(self.orfs**2 * self.sigmas_freqs**-2, axis=1)**-0.5



    def os_sigma(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)
        return 1.0 / np.sqrt(np.sum(self.orfs[mask]**2 / self.sigmas[mask]**2))

    def gw_mean(self):
        return np.array([10**(2.0 * self.params['gw_log10_A'])] * len(self.pairs))

    @functools.lru_cache
    def _tracedot(self, orf, *args):
        ret = np.identity(len(self.Gamma))

        for i, (j,k) in zip(args[::2], args[1::2]):
            ret = np.dot(ret, self.FCFs[i] * self.Gamma * orf(self.psrs[j].pos, self.psrs[k].pos))

        return np.trace(ret)
       
    @functools.lru_cache
    def _tracedot(self, orf, *args):
        ret = np.identity(len(self.Gamma))

        for i, (j,k) in zip(args[::2], args[1::2]):
            ret = np.dot(ret, self.FCFs[i] * self.Gamma)

        return np.trace(ret)

    def gw_corr(self, orf=utils.hd_orf):
        """
        Calculate the covariance matrix between the paired correlations
        This returns
        
        Sigma_{ij, kl} = <rho_{ij}rho_{kl}> - <rho_{ij}><rho_{kl}>,
        
        where rho_{ij} is given in Equation XXX of the NANOGrav 15 year GWB paper.
        """
        Agw = 10 ** self.params['gw_log10_A']

        sigma = np.zeros((len(self.pairs), len(self.pairs)), 'd')

        for ij in tqdm(range(len(self.pairs))):
            i, j = self.pairs[ij]

            for kl in range(ij, len(self.pairs)):
                k, l = self.pairs[kl]
                if ij == kl:
                    o_ij = orf(self.psrs[i].pos, self.psrs[j].pos)
                    term1 = self._tracedot(orf, i, (i, j), j, (j, i))
                    term2 = Agw ** 4 * self._tracedot(orf, i, (i, j), j, (j, i), i, (i, j), j, (j, i)) * o_ij**2
                    sigma[ij, kl] = term1 + term2
                                     

                elif i == k and j != l:
                    o_jl = orf(self.psrs[j].pos, self.psrs[l].pos)
                    o_il = orf(self.psrs[i].pos, self.psrs[l].pos)
                    o_ij = orf(self.psrs[i].pos, self.psrs[j].pos)
                    term1 = Agw ** 2 * self._tracedot(orf, i, (i, j), j, (j, l), l, (l, i)) * o_jl
                    term2 = Agw ** 4 * self._tracedot(orf, i, (i, j), j, (j, i), i, (i, l), l, (l, i)) * o_il * o_ij
                    sigma[ij, kl] = term1 + term2

                    # ijil -> ij il + ii jl + il ji
                    # iCGCj iCGCl -> A^4 (CGCG)^2 + A^2 iCGCj lCGCi + A^4 iCGCj iCGCl

                elif i != k and j == l:
                    o_ik = orf(self.psrs[i].pos, self.psrs[k].pos)
                    o_ij = orf(self.psrs[i].pos, self.psrs[j].pos)
                    o_kj = orf(self.psrs[k].pos, self.psrs[j].pos)
                    term1 = Agw ** 2 * self._tracedot(orf, j, (j, k), k, (k, i), i, (i, j)) * o_ik
                    term2 = Agw ** 4 * self._tracedot(orf, i, (i, j), j, (j, k), k, (k, j), j, (j, i)) * o_ij * o_kj
                    sigma[ij, kl] = term1 + term2
                                     

                    # ijkj -> ij kj + ik jj + ij jk
                    # iCGCj kCGCj -> A^4 (CGCG)^2 + A^2 jCGCk iCGCj + A^4 iCGCj kCGCj

                elif i != k and j != l and i != l and j != k:
                    o_ik = orf(self.psrs[i].pos, self.psrs[k].pos)
                    o_jl = orf(self.psrs[j].pos, self.psrs[l].pos)
                    o_il = orf(self.psrs[i].pos, self.psrs[l].pos)
                    o_kj = orf(self.psrs[k].pos, self.psrs[j].pos)
                    term1 = Agw ** 4 * o_il * o_kj * self._tracedot(orf, i, (i, j), j, (j, k), k, (k, l), l, (l, i))
                    term2 = Agw**4 * o_ik * o_jl * self._tracedot(orf, i, (i, j), j, (j, l), l, (l, k), k, (k, i))
                    sigma[ij, kl] = term1 + term2

                    # ijkl -> ij kl + ik jl + il jk
                    # iCGCj kCGCl -> A^4 (CGCG)^2 + A^4 iCGCj lCGCk + A^4 iCGCj kCGCl
                
                ###
                # NEW CASE, WAS MISSING BEFORE
                ###
                elif j==k:
                    o_ij = orf(self.psrs[i].pos, self.psrs[j].pos)
                    o_jl = orf(self.psrs[j].pos, self.psrs[l].pos)
                    o_il = orf(self.psrs[i].pos, self.psrs[l].pos)
                    term1 = Agw**4 * o_ij * o_jl * self._tracedot(orf, j, (j, i), i, (i, j), j, (j, l), l, (l, j))
                    term2 = Agw**2 * o_il * self._tracedot(orf, j, (j, i), i, (i, l), l, (l, j))
                    sigma[ij, kl] =  term1 + term2
                    

                # apply normalization N_{ij}N_{kl}
                sigma[ij, kl] = sigma[ij, kl] * self.sigmas[ij] ** 2 * self.sigmas[kl] ** 2
                sigma[kl, ij] = sigma[ij, kl]
                
        self.rho_covariance = sigma
        return sigma

    def snr(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        return self.os(orf, sel) / self.os_sigma(orf, sel)
    
    def binned_os(self, orf=utils.hd_orf, nbins=11):
        return makebins(self, nbins)
    
    def correlation_os(self, orf=utils.hd_orf):
        if not hasattr(self, 'rho_covariance'):
            self.gw_corr(orf=orf)
        return full_lstsq_recovery(self, self.rho_covariance)

    def binned_os_with_correlations(self, nbins=11, orf=utils.hd_orf, binedges=None):
        if not hasattr(self, 'rho_covariance'):
            self.gw_corr(orf=orf)
        # get inverse of matrix on rho_{ab} ~ Gamma_{ab} A^2 
        # [as opposed to rho as an estimator of A^2 directly]
        cov_rho = self.rho_covariance

        js = np.linspace(0, self.angles.size + 1, nbins + 1).astype(int)
        angle_idxs = np.argsort(self.angles)
            
        binned_os_vals = []
        binned_os_variances = []
        avg_angles = []
        gamma_bins = []

        
        for ii in range(1, js.size):
            # pick out only indices we want for that bin
            angle_idxs_tmp = angle_idxs[js[ii-1]:js[ii]]
            avg_angles.append(np.mean(self.angles[angle_idxs_tmp]))
            gamma_bin = get_HD_curve(np.mean(self.angles[angle_idxs_tmp]))
            gamma_bins.append(gamma_bin)
            # pull out piece of covariance matrix, invert it
            cov_tmp = cov_rho[angle_idxs_tmp][:, angle_idxs_tmp]

            cov_tmp_inv = np.linalg.inv(cov_tmp)
            # pull out rhos
            rhos_tmp = self.rhos[angle_idxs_tmp]
            # pull out orfs
            orfs_tmp = self.orfs[angle_idxs_tmp]
            # ML solution
            ml_cov = (orfs_tmp.T @ cov_tmp_inv @ orfs_tmp)**-1
            ml_val = gamma_bin * ml_cov * (orfs_tmp.T @ cov_tmp_inv @ rhos_tmp)
            binned_os_vals.append(ml_val)
            
        # Bjk
        binned_estimator_cov = np.zeros((nbins, nbins))

        for ii in range(1, js.size):
            angle_idxs_tmp_ii = angle_idxs[js[ii-1]:js[ii]]
            ubin_ii = self.orfs[angle_idxs_tmp_ii]
            for jj in range(1, js.size):
                angle_idxs_tmp_jj = angle_idxs[js[jj-1]:js[jj]]
                ubin_jj = self.orfs[angle_idxs_tmp_jj]
                
                cov_tmp_ii = cov_rho[angle_idxs_tmp_ii][:, angle_idxs_tmp_ii]
                cov_tmp_jj = cov_rho[angle_idxs_tmp_jj][:, angle_idxs_tmp_jj]
                cov_tmp_ij = cov_rho[angle_idxs_tmp_ii][:, angle_idxs_tmp_jj]
                
                top = ubin_ii.T @ np.linalg.inv(cov_tmp_ii) @ cov_tmp_ij @ np.linalg.inv(cov_tmp_jj) @ ubin_jj
                bot = (ubin_ii.T @ np.linalg.inv(cov_tmp_ii) @ ubin_ii) * (ubin_jj.T @ np.linalg.inv(cov_tmp_jj) @ ubin_jj)
                binned_estimator_cov[ii-1, jj-1] = gamma_bins[ii-1] * gamma_bins[jj-1] * top / bot
        return np.array(avg_angles), np.array(binned_os_vals), binned_estimator_cov
    
    
    def binned_os_with_correlations_alternate_binning(self, orf=utils.hd_orf, binedges=None):
        if not hasattr(self, 'rho_covariance'):
            self.gw_corr(orf=orf)
        # get inverse of matrix on rho_{ab} ~ Gamma_{ab} A^2 
        # [as opposed to rho as an estimator of A^2 directly]
        cov_rho = self.rho_covariance
        binedges = np.array(binedges)
        binned_os_vals = []
        binned_os_variances = []
        avg_angles = []
        gamma_bins = []
        n_angles_per_bin = []
        bin_inds = np.digitize(self.angles, binedges)
        nbins = binedges.size - 1
        
        for ii in range(1, binedges.size):
            # pick out only indices we want for that bin
            mask = bin_inds == ii
            n_angles_per_bin.append(np.sum(mask))
            avg_angles.append(np.mean(self.angles[mask]))
            gamma_bin = get_HD_curve(np.mean(self.angles[mask]))
            gamma_bins.append(gamma_bin)
            # pull out piece of covariance matrix, invert it
            cov_tmp = cov_rho[mask][:, mask]

            cov_tmp_inv = np.linalg.inv(cov_tmp)
            # pull out rhos
            rhos_tmp = self.rhos[mask]
            # pull out orfs
            orfs_tmp = self.orfs[mask]
            # ML solution
            ml_cov = (orfs_tmp.T @ cov_tmp_inv @ orfs_tmp)**-1
            ml_val = gamma_bin * ml_cov * (orfs_tmp.T @ cov_tmp_inv @ rhos_tmp)
            binned_os_vals.append(ml_val)
            
        # Bjk
        binned_estimator_cov = np.zeros((nbins, nbins))
        
        for ii in range(1, binedges.size):
            mask_ii = bin_inds == ii
            ubin_ii = self.orfs[mask_ii]
            for jj in range(1, binedges.size):
                mask_jj = bin_inds == jj
                ubin_jj = self.orfs[mask_jj]
                
                cov_tmp_ii = cov_rho[mask_ii][:, mask_ii]
                cov_tmp_jj = cov_rho[mask_jj][:, mask_jj]
                cov_tmp_ij = cov_rho[mask_ii][:, mask_jj]
                
                top = ubin_ii.T @ np.linalg.inv(cov_tmp_ii) @ cov_tmp_ij @ np.linalg.inv(cov_tmp_jj) @ ubin_jj
                bot = (ubin_ii.T @ np.linalg.inv(cov_tmp_ii) @ ubin_ii) * (ubin_jj.T @ np.linalg.inv(cov_tmp_jj) @ ubin_jj)
                binned_estimator_cov[ii-1, jj-1] = gamma_bins[ii-1] * gamma_bins[jj-1] * top / bot
        return np.array(avg_angles), np.array(binned_os_vals), binned_estimator_cov
    
    def chi2_binned_covariances(self, nbins=11, binedges=None, orf=utils.hd_orf):
        if binedges is not None:
            angles, os_vals, Bjk = self.binned_os_with_correlations_alternate_binning(binedges=binedges)
        else:
            angles, os_vals, Bjk = self.binned_os_with_correlations(nbins, orf)
        resids = (os_vals - get_HD_curve(angles) * 10**(2*self.params['gw_log10_A']))
        chi2 = resids.T @ np.linalg.inv(Bjk) @ resids
        return chi2
    
    def chi2_binned_no_covariances(self, nbins=11, orf=utils.hd_orf):
        out = self.binned_os(orf, nbins)
        angles, os_vals, errors = out[:, 0], out[:, 1], out[:, 2]
        resids = (os_vals - get_HD_curve(angles) * 10**(2*self.params['gw_log10_A']))
        chi2 = resids.T @ np.diag(errors**-2) @ resids
        return chi2
    
    def chi2_covariances(self):
        resids = self.rhos - 10**(2*self.params['gw_log10_A']) * self.orfs
        chi2 = (resids.T @ np.linalg.inv(self.rho_covariance) @ resids)
        return chi2
    
    def chi2_no_covariances(self):
        resids = self.rhos - 10**(2*self.params['gw_log10_A']) * self.orfs
        chi2 = (resids.T @ np.diag(np.diag(self.rho_covariance)**-1) @ resids)
        return chi2
            

def full_lstsq_recovery(os_obj, Sigma=None):
    if Sigma is None:
        print("No covariance matrix supplied, calculating it now...")
        Sigma = os_obj.gw_corr()
    rhos = os_obj.rhos
    design = os_obj.orfs
    # for some reason, I need to do this...
    Sigma_inv = np.linalg.inv(Sigma)

    ml_cov = (design.T @ Sigma_inv @ design)**-1
    ml_val = ml_cov * (design.T @ Sigma_inv @ rhos)
    return ml_val, np.sqrt(ml_cov)

