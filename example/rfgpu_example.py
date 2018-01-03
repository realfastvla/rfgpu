import numpy as np
from numpy.fft import fftshift
import rfgpu

# B-config uv points
uv = np.loadtxt('uv_B.dat')

nbl = uv.shape[0]
nchan = 32
ntime = 128
npix = 1024
upix = npix
vpix = npix/2 + 1

# Set up processing classes
grid = rfgpu.Grid(nbl, nchan, ntime, upix, vpix)
image = rfgpu.Image(npix,npix)

# Data buffers on GPU
vis_raw = rfgpu.GPUArrayComplex((nbl,nchan,ntime))
vis_grid = rfgpu.GPUArrayComplex((upix,vpix))
img_grid = rfgpu.GPUArrayReal((npix,npix))

# Send uv params
grid.set_uv(uv[:,0], uv[:,1]) # u, v in us
grid.set_freq(np.linspace(1000.0,2000.0,nchan)) # freq in MHz
grid.set_shift(np.zeros(nchan,dtype=int)) # dispersion shift per chan in samples
grid.set_cell(80.0) # uv cell size in wavelengths (== 1/FoV(radians))

# Compute gridding transform
grid.compute()

# Generate some random visibility data
vis_raw.data[:] = np.random.randn(*vis_raw.data.shape) \
        + 1.0j*np.random.randn(*vis_raw.data.shape) \
        + 0.5  # Point source at phase center
vis_raw.h2d()  # Send it to GPU memory

# Run gridding on time slice 0 of data array
grid.operate(vis_raw, vis_grid, 0)

# Do FFT
image.operate(vis_grid, img_grid)

# Get image back from GPU
img_grid.d2h()
img_data = fftshift(img_grid.data) # put center pixel in middle of image
