from bz2 import BZ2File
import numpy as np
import matplotlib.pyplot as pp
import os.path as op
import plotutils.autocorr as ac
import triangle as tri

def save_sampler(sampler, outdir):
    with BZ2File(op.join(outdir, 'chain.npy.bz2'), 'w') as out:
        np.save(out, sampler.chain)
    with BZ2File(op.join(outdir, 'lnprob.npy.bz2'), 'w') as out:
        np.save(out, sampler.lnprobability)

def load_sampler(sampler, outdir):
    with BZ2File(op.join(outdir, 'chain.npy.bz2'), 'r') as inp:
        sampler._chain = np.load(inp)
    with BZ2File(op.join(outdir, 'lnprob.npy.bz2'), 'r') as inp:
        sampler._lnprob = np.load(inp)

    return sampler

def plot_posterior(lnprobability, outdir=None):
    pp.figure()
    
    pp.plot(np.mean(lnprobability, axis=0))
    
    pp.xlabel(r'Iteration')
    pp.ylabel(r'$\log \pi$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'posterior.pdf'))

def plot_chain(chain, outdir=None, mean=True):
    pp.figure()

    ndim = chain.shape[-1]
    n = int(np.ceil(np.sqrt(ndim)))

    if mean:
        for i in range(ndim):
            pp.subplot(n,n,i+1)
            pp.plot(np.mean(chain[:,:,i], axis=0))
    else:
        for i in range(ndim):
            pp.subplot(n,n,i+1)
            pp.plot(chain[:,:,i].T)

    if outdir is not None:
        pp.savefig(op.join(outdir, 'chain.pdf'))

def plot_corner(logpost, chain, outdir=None):
    flatchain = chain.reshape((-1, chain.shape[2]))

    tri.corner(flatchain, labels=logpost.pnames, quantiles=[0.05, 0.95])

    if outdir is not None:
        pp.savefig(op.join(outdir, 'corner.pdf'))

def plot_data(logpost, chain, outdir=None):
    pp.figure()

    flatchain = chain.reshape((-1, chain.shape[2]))

    for fmt, p in zip(['-r', '-g', '-b'], np.random.permutation(flatchain)[:3, :]):
        pp.plot(logpost.times, logpost.data_sample(p), fmt, alpha=0.5)

    pp.plot(logpost.times, logpost.intensities, '-k')

    pp.xlabel(r'$t$ (days)')
    pp.ylabel(r'Intensity')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'data-model.pdf'))

def plot_residuals(logpost, chain, outdir=None):
    flatchain = chain.reshape((-1, chain.shape[2]))

    residuals = 0
    for p in flatchain:
        residuals += logpost.detransited_detrended_decorrelated_data(p)
    residuals /= flatchain.shape[0]

    pp.figure()
    pp.plot(logpost.times, residuals, color='k')
    pp.ylabel(r'$\left\langle n(t) \right\rangle$')
    pp.xlabel(r'Time (days)')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'residuals.pdf'))

def plot_all(logpost, chain, lnprobability, outdir=None):
    plot_posterior(lnprobability, outdir=outdir)
    plot_chain(chain, outdir=outdir)
    plot_corner(logpost, chain, outdir=outdir)
    plot_data(logpost, chain, outdir=outdir)
    plot_residuals(logpost, chain, outdir=outdir)
