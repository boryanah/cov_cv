import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def compute_beta_disconnected(power_ln, power_g, power_x, N_mode, k_vals):
    """
    Compute the normalized cross-covariance beta between covariance matrices
    of lognormal and Gaussian power spectra under disconnected approximation.

    Parameters:
    - power_ln: (n_real, n_k) array of P_ln(k) realizations
    - power_g:  (n_real, n_k) array of P_gaussian(k) realizations
    - N_mode:   (n_k,) array of number of modes in each bin
    - k_vals:   (n_k,) array of k values (for plotting)

    Returns:
    - beta: (n_k, n_k) array of cov_xc / var_c
    """
    n_k = len(k_vals)
    P_ln_avg = np.mean(power_ln, axis=0)
    P_g_avg = np.mean(power_g, axis=0)
    P_x_avg = np.mean(power_x, axis=0)

    # Gaussian approximation: disconnected terms only
    cov_xc = np.zeros((n_k, n_k))
    var_c  = np.zeros((n_k, n_k))
    var_x  = np.zeros((n_k, n_k))

    for i in range(n_k):
        for j in range(n_k):
            Nk = N_mode[i]
            Nj = N_mode[j]
            Pi_ln, Pj_ln = P_ln_avg[i], P_ln_avg[j]
            Pi_g,  Pj_g  = P_g_avg[i],  P_g_avg[j]
            Pi_x,  Pj_x  = P_x_avg[i],  P_x_avg[j]
            
            #cov_xc[i, j] += (4 / (power_ln.shape[0] * Nk * Nj)) * Pi_ln * Pj_ln * Pi_g * Pj_g
            cov_xc[i, j] += (4 / (Nk * Nj)) * (Pi_x * Pj_x) ** 2
            var_c[i, j]  += (4 / (Nk * Nj)) * (Pi_g * Pj_g) ** 2
            var_x[i, j]  += (4 / (Nk * Nj)) * (Pi_ln * Pj_ln) ** 2
            
    beta = cov_xc / var_c
    rho = cov_xc / np.sqrt(var_c*var_x)
    return beta, rho

def gaussian_ps_covariance(power_th, N_mode):
    """
    Compute the Gaussian covariance matrix of the power spectrum.
    
    Parameters:
    -----------
    power_th : ndarray of shape (Nk,)
        Theoretical power spectrum values at k bins.
    N_mode : ndarray of shape (Nk,)
        Number of independent Fourier modes per k-bin.

    Returns:
    --------
    cov : ndarray of shape (Nk, Nk)
        Covariance matrix of P(k) under Gaussian assumptions.
    """
    Nk = len(power_th)
    cov = np.zeros((Nk, Nk), dtype=np.float64)

    # Only diagonal terms are non-zero for Gaussian field
    for i in range(Nk):
        if N_mode[i] > 0:
            cov[i, i] = 2.0 * power_th[i]**2 / N_mode[i]
    return cov

# sims
N_sims = 100
data = np.load(f"/pscratch/sd/b/boryanah/cov_cv/lognormal_mocks/lognormal_mock_{0:04d}.npz")
k_avg = data['k_avg']
N_mode = data['N_mode']
bias = data['bias']
b_L = bias - 1.

# fill up with data
powers_tt = np.zeros((N_sims, len(k_avg)))
powers_zz = np.zeros((N_sims, len(k_avg)))
powers_tz = np.zeros((N_sims, len(k_avg)))
for i in range(N_sims):
    data = np.load(f"/pscratch/sd/b/boryanah/cov_cv/lognormal_mocks/lognormal_mock_{i:04d}.npz")
    powers_tt[i] = data['power_ln']
    powers_tz[i] = data['power_ln_1cb'] + b_L*data['power_ln_delta']

    data = np.load(f"/pscratch/sd/b/boryanah/cov_cv/zeldovich/adv_power_{i:04d}.npz", allow_pickle=True)
    power = data['power']
    powers_zz[i] = power.item()['1cb_1cb']+2.*b_L*power.item()['1cb_delta']+b_L**2*power.item()['delta_delta']

# compute beta
beta, rho = compute_beta_disconnected(powers_tt, powers_zz, powers_tz, N_mode, k_avg)
    
# linear theory
pk_k = data['k_lin']
pk_vals = data['pk_lin']
pk_interp = interp1d(pk_k, pk_vals, kind='linear', bounds_error=False, fill_value=0.0)
power_th = bias**2*pk_interp(k_avg)
cov_th = gaussian_ps_covariance(power_th, N_mode)

# zeldovich theory
N_theo = 1000
powers_th = np.zeros((N_theo, len(k_avg)))
for i in range(N_theo):
    data = np.load(f"/pscratch/sd/b/boryanah/cov_cv/zeldovich/adv_power_{i:04d}.npz", allow_pickle=True)
    power = data['power']
    powers_th[i] = power.item()['1cb_1cb']+2.*b_L*power.item()['1cb_delta']+b_L**2*power.item()['delta_delta']

# control variates
#beta[:, :] = 1. # TESTING
X = np.cov(powers_tt.T)
C = np.cov(powers_zz.T)
#mu = cov_th # Gaussian version
mu = np.cov(powers_th.T)
Y = X - beta*(C - mu)

if 0:#True:
    plt.errorbar(k_avg, power_th, label='theory')
    #plt.plot(k_avg, powers_ic.T, alpha=0.5)
    plt.errorbar(k_avg, np.mean(powers_zz, axis=0), np.std(powers_zz, axis=0)/np.sqrt(N_sims), label='zz')
    plt.errorbar(k_avg, np.mean(powers_tz, axis=0), label='tz')
    plt.errorbar(k_avg, np.mean(powers_tt, axis=0), label='tt')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig("power.png")
    plt.show()
    quit()

def cov2corr(cov):
    s = np.sqrt(np.diag(cov))
    return cov/s[None, :]/s[:, None]

def plot_kkpr(kkpr, vmin=None, vmax=None):
    plt.imshow(kkpr, origin='lower', extent=[k_avg[0], k_avg[-1], k_avg[0], k_avg[-1]],
               aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.xlabel(r'$k$ [$h/$Mpc]')
    plt.ylabel(r"$k'$ [$h/$Mpc]")
    plt.tight_layout()

if 0:#True:
    plt.figure(1, figsize=(6,5))
    plot_kkpr(mu)
    plt.colorbar(label=r'$\mu(k, k^\prime)$')
    plt.savefig("mu.png")

    plt.figure(2, figsize=(6,5))
    plot_kkpr(C)
    plt.colorbar(label=r'$C(k, k^\prime)$')
    plt.savefig("C.png")
    
    plt.figure(3, figsize=(6,5))
    plot_kkpr(C-mu)
    plt.colorbar(label=r'$C-\mu(k, k^\prime)$')
    plt.savefig("C-mu.png")
    plt.show()
    quit()

# Plot beta matrix
plt.figure(1, figsize=(6,5))
plot_kkpr(beta)
plt.colorbar(label=r'$\beta(k, k^\prime)$')
plt.savefig("beta.png")

plt.figure(5, figsize=(6,5))
plot_kkpr(rho)
plt.colorbar(label=r'$\rho(k, k^\prime)$')
plt.savefig("rho.png")

plt.figure(6, figsize=(6,5))
plot_kkpr(1./(1-rho**2))
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label=r'$1/(1-\rho^2)(k, k^\prime)$')
plt.savefig("N_sim_gain.png")

plt.figure(2, figsize=(6,5))
plot_kkpr(cov2corr(X), vmin=0, vmax=1)
plt.colorbar(label=r'$X(k, k^\prime)$')
plt.savefig("X.png")

plt.figure(3, figsize=(6,5))
plot_kkpr(cov2corr(Y), vmin=0, vmax=1)
plt.colorbar(label=r'$Y(k, k^\prime)$')
plt.savefig("Y.png")

plt.figure(4, figsize=(6,5))
plot_kkpr(cov2corr(X)-cov2corr(Y))
plt.colorbar(label=r'$X-Y(k, k^\prime)$')
plt.savefig("X-Y.png")
plt.show()
