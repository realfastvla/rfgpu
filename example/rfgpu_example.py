import numpy as np
from numpy.fft import fftshift
import rfgpu

# Select which GPU to use; this must be done before calling any other
# rfgpu functions.
rfgpu.cudaSetDevice(0)

# B-config uv points
uv = np.loadtxt('uv_B_signed.dat')

nbl = uv.shape[0]
nchan = 128
ntime = 256
xpix = 1024
ypix = 1024
upix = xpix
vpix = ypix/2 + 1
freq = np.linspace(1000.0,2000.0,nchan)
dt = 5e-3
ddm = 1.0

def dm_delay(dm,f1,f2):
    # Freq in MHz, delay in s
    return (dm/0.000241)*(1.0/(f1*f1) - 1.0/(f2*f2))

# Set up shift arrays
max_dm = (ntime*dt/2.0) / dm_delay(1.0,freq.min(),freq.max())
ndm = int(max_dm/ddm)
shifts = np.zeros((ndm,nchan),dtype=int)
for idm in range(ndm):
    shifts[idm,:] = dm_delay(idm*ddm, freq, freq.max())/dt

# Set up processing classes
grid = rfgpu.Grid(nbl, nchan, ntime, upix, vpix)
image = rfgpu.Image(xpix,ypix)

# Set up which statistics we want Image to compute during each call
# to Image.stats()
image.add_stat('rms')
image.add_stat('max')

# Data buffers on GPU
vis_raw = rfgpu.GPUArrayComplex((nbl,nchan,ntime))
vis_grid = rfgpu.GPUArrayComplex((upix,vpix))
img_grid = rfgpu.GPUArrayReal((xpix,ypix))

# Send uv params
grid.set_uv(uv[:,0], uv[:,1]) # u, v in us
grid.set_freq(freq) # freq in MHz
grid.set_shift(shifts[0]) # dispersion shift per chan in samples
grid.set_cell(80.0) # uv cell size in wavelengths (== 1/FoV(radians))

# Compute gridding transform
grid.compute()

# Generate some visibility data
vis_raw.data[:] = np.random.randn(*vis_raw.data.shape) \
        + 1.0j*np.random.randn(*vis_raw.data.shape) \
        + 0.0  # Point source at phase center

# Add a point source somewhere else, only in single time slice
dx = 10.0/60.0 * np.pi/180.0
dy = 3.0/60.0 * np.pi/180.0
ul = np.outer(uv[:,0],freq)
vl = np.outer(uv[:,1],freq)
vis = 0.5 * np.exp(-2.0j*np.pi*(dx*ul + dy*vl))
vis_raw.data[:,:,15] += vis

# put another one in, with dispersion
for ichan in range(nchan):
    vis_raw.data[:,ichan,40+shifts[ndm/2,ichan]] += vis[:,ichan]

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

# Image time slice 1, this should have two point sources
grid.operate(vis_raw, vis_grid, 15)
image.operate(vis_grid, img_grid)
img_grid.d2h()
img_data1 = fftshift(img_grid.data)

# Image all dm/time slices, get back rms and max value for each
nds = 4 # number of downsamples
img_rms = np.zeros((nds,ndm,ntime/2))
img_max = np.zeros((nds,ndm,ntime/2))
for ids in range(nds):
    for idm in range(ndm):
        grid.set_shift(shifts[idm]>>ids)
        for itime in range((ntime/2)>>ids):
            grid.operate(vis_raw,vis_grid,itime)
            image.operate(vis_grid,img_grid)
            s = image.stats(img_grid)
            img_rms[ids,idm,itime] = s['rms']
            img_max[ids,idm,itime] = s['max']
    grid.downsample(vis_raw)
