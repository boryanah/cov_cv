import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fftfreq, rfftfreq
from scipy.fft import rfftn, irfftn, fftn, ifftn

from classy import Class
from abacusnbody.analysis.power_spectrum import calc_power, calc_pk_from_deltak, get_W_compensated
from abacusnbody.analysis.tsc import tsc_parallel

from nbodykit.lab import LinearMesh, cosmology, LogNormalCatalog
from nbodykit import setup_logging

nmesh = 256//2
Lbox = 2000.0  # Mpc/h
bias = 2.2
z_ic = 99
z_mock = 0.5
ngal_mean = 1.e-3 # (Mpc/h)^-3

interlaced = False
compensated = True
paste = 'TSC'
nbins_mu = 1
logk = False
k_hMpc_max = np.pi * nmesh / Lbox
nbins_k = nmesh // 2
k_bin_edges = np.linspace(1.e-6, k_hMpc_max, nbins_k)
mu_bin_edges = np.array([0., 1.])

cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=z_mock, transfer='EisensteinHu')

from numba import njit
                
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

                # Lexicographic ordering to avoid overwriting
                if (i > i_conj or
                    (i == i_conj and j > j_conj) or
                    (i == i_conj and j == j_conj and k >= k_conj)):
                    delta_k[i, j, k] = np.conj(delta_k[i_conj, j_conj, k_conj])
                    
def generate_gaussian_field_nbodykit(pk, nmesh, boxsize, seed):
    # Create a 3D Gaussian density field
    mesh = LinearMesh(pk, Nmesh=nmesh, BoxSize=boxsize, seed=seed)
    delta = mesh.compute().preview()
    delta -= 1.
    return delta

def generate_gaussian_field(pk_interp, nmesh, boxsize, seed=None):
    """Generate a Gaussian random field with power spectrum pk_interp(k). rfftn fails"""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()        
    # Define fundamental frequency
    dk = 2 * np.pi / boxsize

    # Create Fourier-space grid
    kx = fftfreq(nmesh, d=1./nmesh) * dk
    ky = fftfreq(nmesh, d=1./nmesh) * dk
    kz = fftfreq(nmesh, d=1./nmesh) * dk
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k_mag = np.sqrt(k2)
    
    # Power spectrum P(k) on grid
    pk = pk_interp(k_mag)
    pk[k_mag == 0] = 0.0  # Zero mode has no power
    
    # Sample Rayleigh amplitude: A ~ Rayleigh(1)
    amplitude = rng.rayleigh(scale=1.0, size=k_mag.shape)
    
    # Random phase θ ∈ [0, 2π]
    phase = rng.uniform(0, 2*np.pi, size=k_mag.shape)

    # Construct complex field: delta_k = A * exp(iθ) * sqrt(P(k)/V)
    V = boxsize**3
    delta_k = amplitude * np.exp(1j * phase) * np.sqrt(pk / V / 2.)

    # Enforce Hermitian symmetry manually for inverse Fourier to be real
    enforce_hermitian_symmetry(delta_k)
    delta_k *= nmesh**3
    delta_k[0, 0, 0] = 0.0  # Remove DC component
    delta_k = delta_k.astype(np.complex64)
    
    delta_x = np.real(ifftn(delta_k, s=(nmesh, nmesh, nmesh)))
    delta_x = delta_x.astype(np.float32)
    
    # save the displacement fields as well
    k2[0, 0, 0] = 1.
    disp_x = 1j * kx/k2 * delta_k
    disp_x = np.real(ifftn(disp_x, s=(nmesh, nmesh, nmesh)))
    disp_y = 1j * ky/k2 * delta_k
    disp_y = np.real(ifftn(disp_y, s=(nmesh, nmesh, nmesh)))
    disp_z = 1j * kz/k2 * delta_k
    disp_z = np.real(ifftn(disp_z, s=(nmesh, nmesh, nmesh)))
    return delta_x, disp_x, disp_y, disp_z

def generate_gaussian_field_oldest(pk_interp, nmesh, boxsize, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    kf = 2 * np.pi / boxsize
    kx = fftfreq(nmesh, d=1./nmesh).astype(np.float32) * kf
    ky = fftfreq(nmesh, d=1./nmesh).astype(np.float32) * kf
    kz = fftfreq(nmesh, d=1./nmesh).astype(np.float32) * kf
    #kz = rfftfreq(nmesh, d=1./nmesh).astype(np.float32) * kf
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    kk = np.sqrt(kx**2 + ky**2 + kz**2)
    
    #pk = pk_interp(kk.flatten()).reshape(nmesh, nmesh, nmesh//2+1).astype(np.float32)
    pk = pk_interp(kk.flatten()).reshape(nmesh, nmesh, nmesh).astype(np.float32)
    pk[kk == 0] = 0.0

    #noise = rng.normal(0, 1, (nmesh, nmesh, nmesh//2+1)) + 1j * rng.normal(0, 1, (nmesh, nmesh, nmesh//2+1))
    noise = rng.normal(0, 1, (nmesh, nmesh, nmesh)) + 1j * rng.normal(0, 1, (nmesh, nmesh, nmesh))
    noise = noise.astype(np.complex64)
    delta_k = noise * np.sqrt(pk / 2.0 / boxsize**3 ) * nmesh**3 
    delta_k[0, 0, 0] = 0.0

    #delta_x = np.real(irfftn(delta_k))
    delta_x = np.real(ifftn(delta_k))
    return delta_x

def generate_gaussian_field_old(pk_interp, nmesh, boxsize):
    """this is fixed ICs (note issues)"""
    kf = 2 * np.pi / boxsize  # fundamental mode
    kx = fftfreq(nmesh, d=1./nmesh) * kf
    ky = fftfreq(nmesh, d=1./nmesh) * kf
    kz = rfftfreq(nmesh, d=1./nmesh) * kf
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    kk = np.sqrt(kx**2 + ky**2 + kz**2)
    pk_vals = pk_interp(kk)
    pk_vals[kk == 0] = 0.0  # zero mode

    # Random phase and fixed amplitude
    phase = np.random.uniform(0, 2 * np.pi, size=kk.shape)
    amplitude = np.sqrt(pk_vals / boxsize**3) * nmesh**3
    delta_k = amplitude * (np.cos(phase) + 1j * np.sin(phase))

    # Hermitian symmetry to get real field after irfftn
    delta_k[0, 0, 0] = 0.0  # real mean zero
    delta_x = np.real(irfftn(delta_k, s=(nmesh, nmesh, nmesh)))
    return delta_x
                                                    
def generate_lognormal_field(delta_x, bias, mean=0.0):
    """
    var = np.var(delta_x)
    ln_field = np.exp(delta_x*np.sqrt(var) - var/2.0) - 1.0
    print("mean", np.mean(ln_field))
    ln_field *= bias
    """
    lagr_bias = bias - 1.
    ln_field = np.exp(delta_x*lagr_bias)
    ln_field /= np.mean(ln_field, dtype=np.float64)
    ln_field -= 1.
    return ln_field

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

def compute_power_field(field, field2=None):
    field_fft = rfftn(field)/field.size
    if field2 is not None:
        field2_fft = rfftn(field2)/field2.size
    else:
        field2_fft = None
    power = calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
    return power['k_avg'], power['power'], power['N_mode']

def main():
    want_nbodykit = False
    
    # mine:
    pk_k, pk_vals = generate_linear_pk(z=z_mock)
    pk_interp = interp1d(pk_k, pk_vals, kind='linear', bounds_error=False, fill_value=0.0)

    if want_nbodykit:
        # nbodykit:
        pk_vals = Plin(pk_k)
    
    for i in range(1000, 2000):
        print(i)
        seed = 300+i

        if want_nbodykit:
            # nbodykit:
            #delta_x = generate_gaussian_field_nbodykit(pk_interp, nmesh, Lbox, seed)
            delta_x = generate_gaussian_field_nbodykit(Plin, nmesh, Lbox, seed)
        else:
            # mine:
            delta_x, disp_x, disp_y, disp_z = generate_gaussian_field(pk_interp, nmesh, Lbox, seed) # works! lehman says that's zd too
            #delta_x = generate_gaussian_field_oldest(pk_interp, nmesh, Lbox, seed) # erm weird amplitude?

        if want_nbodykit:
            # nbodykit:
            ln_field = LogNormalCatalog(Plin=Plin, nbar=ngal_mean, BoxSize=Lbox, Nmesh=nmesh, bias=bias, seed=seed).to_mesh(compensated=True, window='cic', position='Position').compute().preview().astype(np.float32)-1.
            #ln_field = np.load("ln_field_nbodykit_0000.npy")-1.
        else:
            # mine:
            ln_field = generate_lognormal_field(delta_x, bias=bias)
            positions = sample_galaxies(ln_field, disp_x, disp_y, disp_z, ngal_mean=ngal_mean, boxsize=Lbox, seed=seed)
            positions = positions.astype(np.float32)
            print(positions.dtype)
            ln_field[:, :, :] = 0.
            print(ln_field.dtype)
            ln_field = tsc_parallel(positions, ln_field, Lbox) 
            ln_field /= np.mean(ln_field)
            ln_field -= 1.
            W = get_W_compensated(Lbox, nmesh, paste, interlaced=False)
            ln_field = irfftn(rfftn(ln_field)/ W[:, np.newaxis, np.newaxis] / W[np.newaxis, :, np.newaxis] / W[np.newaxis, np.newaxis, : (nmesh // 2 + 1)])
            print(ln_field.dtype)
            ln_field = ln_field.astype(np.float32)

            
        # nbodykit:
        #positions = np.load("pos_nbodykit_0000.npy")
        print("means", delta_x.mean(), ln_field.mean())

        #k_avg, power = compute_power_pos(positions)
        k_avg, power, N_mode = compute_power_field(ln_field, field2=None)
        k_avg, power_ic, N_mode = compute_power_field(bias*delta_x, field2=None)
        k_avg, power_x, N_mode = compute_power_field(bias*delta_x, field2=ln_field)

        np.savez(f"/mnt/gosling1/boryanah/lognormal_mocks/delta_{i:04d}.npz", ic_delta=delta_x, ln_field=ln_field, k_avg=k_avg, power=power, power_ic=power_ic, power_x=power_x, pk_k=pk_k, pk_vals=pk_vals, z_mock=z_mock, bias=bias, N_mode=N_mode)
main()
