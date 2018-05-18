from pymc3 import hpd as HPD
import matplotlib.pyplot as pl
from numpy import array as ARRAY

def modified_forest_plot(trace, varnames, figsize=None):
    n_vars = len(varnames)
    if figsize is None:
        figsize = (5*n_vars, 5)
    f, ax = pl.subplots(ncols=n_vars, figsize=figsize, sharey=True)
    n_chains = trace.nchains
    if isinstance(ax, np.ndarray):
        axs = ax.ravel()
    else:
        axs = ARRAY([ax]).ravel()    
    for axi, var in zip(axs, varnames):
        var_trace = trace[var]
        chain_length = int(var_trace.shape[0] / n_chains)
        start_idx = 0
        alpha=0.05
        iqr=0.5
        n_provs = var_trace.shape[1]
        chain_separation = 0.1
        axi.axvline(linestyle='--', zorder=0)
        axi.set_ylim(n_provs+1, 0)
        
        axi.set_yticks([i for i in range(1, n_provs+1)])
        axi.set_yticklabels([prov_mapping[i] for i in range(n_provs)])
        chain_ys = np.array([chain_separation * i for i in range(1, n_chains+1)])
        for provi in range(n_provs):
            chain_ys_provi = chain_ys - (n_chains * chain_separation)/2 - 0.05 + provi+1
            for chain_idx in range(0, n_chains):
                start_idx = chain_idx * chain_length
                end_idx = (chain_idx+1) * chain_length
                chain = var_trace[start_idx: end_idx, provi]
                chain_mean = chain.mean()
                chain_ci_alpha = HPD(chain, alpha=alpha)
                chain_iqr = HPD(chain, alpha=iqr)
                chain_yi = chain_ys_provi[chain_idx]
                axi.hlines(chain_yi, xmin=chain_ci_alpha[0], xmax=chain_ci_alpha[1],
                           linewidth=0.5)
                axi.hlines(chain_yi, xmin=chain_iqr[0], xmax=chain_iqr[1], linewidth=3);
                axi.scatter(chain_mean, chain_yi, s=30, color='k')
        axi.set_title(var)
        axi.grid(axis='y', linestyle='-.', color='gray')
    f.subplots_adjust(wspace=0.025)
    f.suptitle('95% Credible Intervals', y=0.98);
    
    return f, axs