"""
could also save the linear version for comparison
"""
import gc
import time

import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fftfreq, rfftfreq
from scipy.fft import rfftn, irfftn, fftn, ifftn
from numba import njit

from classy import Class
from abacusnbody.analysis.power_spectrum import calc_power, calc_pk_from_deltak, get_W_compensated
from abacusnbody.analysis.tsc import tsc_parallel

nmesh = 128*4
Lbox = 2000.0  # Mpc/h
bias = 2.2
z_mock = 0.5
ngal_mean = 1.e-3 # (Mpc/h)^-3

interlaced = False
compensated = True
paste = 'TSC'
logk = False
k_hMpc_max = np.pi * nmesh / Lbox
nbins_k = nmesh // 2
k_bin_edges = np.linspace(0., k_hMpc_max, nbins_k)
mu_bin_edges = np.array([0., 1.])
                
@njit
def enforce_hermitian_symmetry(delta_k):
    nx, ny, nz = delta_k.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute negative indices with periodic wrapping
                i_conj = (-i) % nx
                j_conj = (-j) % ny
                k_conj = (-k) % nz

                # Lexicographic ordering to avoid overwriting (one of k[i], k[j], k[k] is negative
                if (i > i_conj or
                    (i == i_conj and j > j_conj) or
                    (i == i_conj and j == j_conj and k >= k_conj)):
                    delta_k[i, j, k] = np.conj(delta_k[i_conj, j_conj, k_conj])
                    
def generate_gaussian_field(kx, ky, kz, k2, pk_factor, nmesh, boxsize, seed=None):
    """Generate a Gaussian random field with power spectrum pk_interp(k). rfftn fails"""
    t = time.time()
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    # Sample Rayleigh amplitude: A ~ Rayleigh(1)
    amplitude = rng.rayleigh(scale=1.0, size=pk_factor.shape).astype(np.float32)
    
    # Random phase θ ∈ [0, 2π]
    phase = rng.uniform(0, 2*np.pi, size=pk_factor.shape).astype(np.float32)

    # Construct complex field: delta_k = A * exp(iθ) * sqrt(P(k)/V/2)
    delta_k = amplitude * np.exp(1j * phase) * pk_factor
    print("generating", time.time()-t)
    
    # Enforce Hermitian symmetry manually for inverse Fourier to be real
    t = time.time()
    enforce_hermitian_symmetry(delta_k)
    print("enforcing", time.time()-t)
    delta_k *= nmesh**3
    delta_k[0, 0, 0] = 0.0  # Remove DC component
    
    delta_ic = np.real(ifftn(delta_k, s=(nmesh, nmesh, nmesh), workers=-1))

    t = time.time()
    # save the displacement fields as well
    disp_x = 1j * kx/k2 * delta_k
    disp_x = np.real(ifftn(disp_x, s=(nmesh, nmesh, nmesh), workers=-1))
    disp_y = 1j * ky/k2 * delta_k
    disp_y = np.real(ifftn(disp_y, s=(nmesh, nmesh, nmesh), workers=-1))
    disp_z = 1j * kz/k2 * delta_k
    disp_z = np.real(ifftn(disp_z, s=(nmesh, nmesh, nmesh), workers=-1))
    print("displacements", time.time()-t)
    return delta_ic, disp_x, disp_y, disp_z

def generate_lognormal_field(delta_ic, bias, mean=0.0):
    lagr_bias = np.float32(bias - 1.)
    delta_ln = np.exp(delta_ic*lagr_bias)
    delta_ln /= np.mean(delta_ln, dtype=np.float64)
    delta_ln -= 1.
    return delta_ln

def sample_galaxies(ln_field, disp_x, disp_y, disp_z, ngal_mean, boxsize, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()        
    
    cell_volume = (boxsize / ln_field.shape[0])**3
    lam = ln_field + 1.0
    lam *= ngal_mean * cell_volume / lam.mean()  # normalize to desired density
    ngal = np.random.poisson(lam)
    Ngal = np.sum(ngal)
    
    positions = np.zeros((Ngal, 3), dtype=np.float32)

    ngrid = ngal.shape[0]
    indices = np.argwhere(ngal > 0)  # shape (M, 3)
    counts = ngal[ngal > 0]          # shape (M,)

    # Repeat voxel indices according to counts
    repeated_indices = np.repeat(indices, counts, axis=0)  # shape (N, 3)
    disp_x = np.repeat(disp_x[indices[:, 0], indices[:, 1], indices[:, 2]], counts, axis=0)
    disp_y = np.repeat(disp_y[indices[:, 0], indices[:, 1], indices[:, 2]], counts, axis=0)
    disp_z = np.repeat(disp_z[indices[:, 0], indices[:, 1], indices[:, 2]], counts, axis=0)
    
    # Generate uniform random positions inside each unit cube
    offsets = rng.random((len(repeated_indices), 3))

    # Add voxel indices and scale to box units
    positions = (repeated_indices + offsets) * (boxsize / ngrid)

    # add the displacements
    positions[:, 0] += disp_x
    positions[:, 1] += disp_y
    positions[:, 2] += disp_z
    positions %= boxsize
    
    return positions

def generate_linear_pk(z=99, kmin=1e-4, kmax=10, nk=2048, h=0.6736):
    cosmo = Class()
    cosmo.set({
        'output': 'mPk',
        'P_k_max_h/Mpc': kmax,
        'z_pk': z,
        'h': h,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau_reio': 0.0544,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06
    })
    cosmo.compute()

    # k in h/Mpc for your box
    k_h = np.logspace(np.log10(kmin), np.log10(kmax), nk)  # [h/Mpc]

    # Convert to 1/Mpc for CLASS
    k_class = k_h * h  # [1/Mpc]
    pk_class = np.array([cosmo.pk(k, z) for k in k_class])  # [Mpc^3]
    
    # Convert to h^-3 Mpc^3
    pk_h = pk_class * h**3  # [(Mpc/h)^3]
    
    cosmo.struct_cleanup()
    cosmo.empty()
    return k_h, pk_h  # both in h-units now                                                

def compute_power_pos(pos):
    # grid_field: 3D array (overdensity or lognormal field)
    power = calc_power(
        pos,
        Lbox,
        k_bin_edges,
        mu_bin_edges,
        logk=logk,
        paste=paste,
        nmesh=nmesh,
        compensated=compensated,
        interlaced=interlaced,
    )
    return power['k_avg'], power['power']

def compute_power_field(field, field2=None, W=None, W2=None):
    field_fft = rfftn(field, workers=-1)/field.size
    if field2 is not None:
        field2_fft = rfftn(field2)/field2.size
    else:
        field2_fft = None
    if W is not None:
        field_fft /= W[:, np.newaxis, np.newaxis] * W[np.newaxis, :, np.newaxis] * W[np.newaxis, np.newaxis, : (nmesh // 2 + 1)]
    if W2 is not None and field2 is not None:
        field2_fft /= W2[:, np.newaxis, np.newaxis] * W2[np.newaxis, :, np.newaxis] * W2[np.newaxis, np.newaxis, : (nmesh // 2 + 1)]
    power = calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
    return power['k_avg'], power['power'], power['N_mode']

def main():
    want_lognormal = False
    
    k_lin, pk_lin = generate_linear_pk(z=z_mock)
    pk_interp = interp1d(k_lin, pk_lin, kind='linear', bounds_error=False, fill_value=0.0)

    # Define fundamental frequency
    dk = 2 * np.pi / Lbox

    # Create Fourier-space grid
    kx = (fftfreq(nmesh, d=1./nmesh) * dk).astype(np.float32)
    ky = (fftfreq(nmesh, d=1./nmesh) * dk).astype(np.float32)
    kz = (fftfreq(nmesh, d=1./nmesh) * dk).astype(np.float32)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.
    k_mag = np.sqrt(k2)
    
    # Power spectrum P(k) on grid
    pk = pk_interp(k_mag)
    pk[0, 0, 0] = 0.0  # Zero mode has no power
    pk_factor = np.sqrt(pk / Lbox**3 / 2.).astype(np.float32)
    del k_mag, pk
    gc.collect()
    
    W = get_W_compensated(Lbox, nmesh, paste, interlaced=False)
    
    for i_sim in range(83, 1000):
        print(i_sim)
        seed = 300+i_sim
        delta_ic, disp_x, disp_y, disp_z = generate_gaussian_field(kx, ky, kz, k2, pk_factor, nmesh, Lbox, seed)
        #np.savez(f"/pscratch/sd/b/boryanah/cov_cv/zeldovich/zeldovich_{i_sim:04d}.npz", delta_ic=delta_ic, disp_x=disp_x, disp_y=disp_y, disp_z=disp_z, k_lin=k_lin, pk_lin=pk_lin, z_mock=z_mock, nmesh=nmesh, Lbox=Lbox)

        t = time.time()
        # I think bc displacements is actually at kx, k^2 which is not the centers!!!
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(nmesh, dtype=np.float32) * Lbox / nmesh,
            np.arange(nmesh, dtype=np.float32) * Lbox / nmesh,
            np.arange(nmesh, dtype=np.float32) * Lbox / nmesh,
            indexing='ij',
        )
        disp_pos = np.zeros((nmesh*nmesh*nmesh, 3), dtype=np.float32)
        disp_pos[:, 0] = grid_x.flatten() + disp_x.flatten()
        disp_pos[:, 1] = grid_y.flatten() + disp_y.flatten()
        disp_pos[:, 2] = grid_z.flatten() + disp_z.flatten()
        print("grid", time.time()-t)

        t = time.time()
        adv_1cb = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
        adv_delta = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
        tsc_parallel(disp_pos, adv_1cb, Lbox)
        tsc_parallel(disp_pos, adv_delta, Lbox, weights=delta_ic.flatten())
        del disp_pos, grid_x, grid_y, grid_z
        gc.collect()
        
        mean_1cb = np.mean(adv_1cb, dtype=np.float64)
        adv_1cb /= mean_1cb
        adv_1cb -= 1.
        adv_delta /= mean_1cb
        print("tsc", time.time()-t)
        
        t = time.time()
        power = {}
        #k_avg, power['ic_ic'], N_mode = compute_power_field(delta_ic, field2=None, W=None)
        k_avg, power['1cb_1cb'], N_mode = compute_power_field(adv_1cb, field2=None, W=W)
        k_avg, power['delta_delta'], N_mode = compute_power_field(adv_delta, field2=None, W=W)
        k_avg, power['1cb_delta'], N_mode = compute_power_field(adv_1cb, field2=adv_delta, W=W, W2=W)
        power['delta_1cb'] = power['1cb_delta']

        np.savez(f"/pscratch/sd/b/boryanah/cov_cv/zeldovich/adv_power_{i_sim:04d}.npz", power=power, k_avg=k_avg, N_mode=N_mode, k_lin=k_lin, pk_lin=pk_lin, z_mock=z_mock, nmesh=nmesh, Lbox=Lbox)
        print("power zd", time.time()-t)
        # note that we will also need the cross power spectrum for say 1000 or 500 sims!!!!!!!!!!!!!!!!!!

        if want_lognormal or i_sim < 100:
            t = time.time()
            delta_ln = generate_lognormal_field(delta_ic, bias=bias)
            pos_ln = sample_galaxies(delta_ln, disp_x, disp_y, disp_z, ngal_mean=ngal_mean, boxsize=Lbox, seed=seed)
            pos_ln = pos_ln.astype(np.float32)
            delta_ln[:, :, :] = 0.
            delta_ln = tsc_parallel(pos_ln, delta_ln, Lbox) 
            delta_ln /= np.mean(delta_ln)
            delta_ln -= 1.
            print("lognormal", time.time()-t)

            t = time.time()
            # compute power spectra
            k_avg, power_ln, N_mode = compute_power_field(delta_ln, field2=None, W=W)
            #k_avg, power_ln_ic, N_mode = compute_power_field(delta_ln, field2=delta_ic, W=W, W2=None)
            k_avg, power_ln_1cb, N_mode = compute_power_field(delta_ln, field2=adv_1cb, W=W, W2=W)
            k_avg, power_ln_delta, N_mode = compute_power_field(delta_ln, field2=adv_delta, W=W, W2=W)
            
            np.savez(f"/pscratch/sd/b/boryanah/cov_cv/lognormal_mocks/lognormal_mock_{i_sim:04d}.npz", k_avg=k_avg, N_mode=N_mode, power_ln=power_ln, power_ln_1cb=power_ln_1cb, power_ln_delta=power_ln_delta, bias=bias, k_lin=k_lin, pk_lin=pk_lin, z_mock=z_mock, nmesh=nmesh, Lbox=Lbox)
            print("power ln", time.time()-t)
            del delta_ln, pos_ln

        del delta_ic, adv_1cb, adv_delta, disp_x, disp_y, disp_z
        gc.collect()

main()
