import numpy as np
from numpy.fft import fftshift
import rfgpu

# B-config uv points
uv = np.loadtxt('uv_B_signed.dat')

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
freq = np.linspace(1000.0,2000.0,nchan)
grid.set_uv(uv[:,0], uv[:,1]) # u, v in us
grid.set_freq(freq) # freq in MHz
grid.set_shift(np.zeros(nchan,dtype=int)) # dispersion shift per chan in samples
grid.set_cell(80.0) # uv cell size in wavelengths (== 1/FoV(radians))

# Compute gridding transform
grid.compute()

# Generate some visibility data
vis_raw.data[:] = np.random.randn(*vis_raw.data.shape) \
        + 1.0j*np.random.randn(*vis_raw.data.shape) \
        + 0.0  # Point source at phase center

# Add a point source somewhere else, only in time slice 1
dx = 10.0/60.0 * np.pi/180.0
dy = 3.0/60.0 * np.pi/180.0
ul = np.outer(uv[:,0],freq)
vl = np.outer(uv[:,1],freq)
vis_raw.data[:,:,1] += 0.5 * np.exp(-2.0j*np.pi*(dx*ul + dy*vl))

vis_raw.h2d()  # Send it to GPU memory

# Conjugate vis data as needed (on GPU); only do once 
grid.conjugate(vis_raw)

# Run gridding on time slice 0 of data array
grid.operate(vis_raw, vis_grid, 0)

# Do FFT
image.operate(vis_grid, img_grid)

# Get image back from GPU
img_grid.d2h()
img_data0 = fftshift(img_grid.data) # put center pixel in middle of image
print('img 0: max {0}, std {1}, peak {2}'
      .format(img_data0.max(), img_data0.std(), np.where(img_data0 == img_data0.max())))

# Image time slice 1, this should have two point sources
grid.operate(vis_raw, vis_grid, 1)
image.operate(vis_grid, img_grid)
img_grid.d2h()
img_data1 = fftshift(img_grid.data)
print('img 1: max {0}, std {1}, peak {2}'
      .format(img_data1.max(), img_data1.std(), np.where(img_data1 == img_data1.max())))

# Image all time slices, get back rms and max value for each
img_rms = np.zeros(ntime)
img_max = np.zeros(ntime)
for i in range(ntime):
    grid.operate(vis_raw,vis_grid,i)
    image.operate(vis_grid,img_grid)
    s = image.stats(img_grid)
    img_rms[i] = np.sqrt(s[0])/npix
    img_max[i] = s[1]
    print('img {0}: max {1}, std {2}'
          .format(i, img_max[i], img_rms[i]))
