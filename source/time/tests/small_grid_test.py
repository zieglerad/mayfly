import numpy as np
import matplotlib.pyplot as plt 
import time_domain.func as func 
from skimage import feature
from skimage import transform
from skimage import morphology
from scipy import stats
array_radius=10e-02
array_sep=6

Rant,Tant=func.array_geo(array_radius,array_sep)


Flo=25e9
Fsrc=np.array([24.95e9])
Fsamp=200e6
Nsamp=8192
As=np.array([1e-6])

s=func.source_signal(Fsrc,Flo,Fsamp,Nsamp,As)

Rsrc=[2e-2]
Tsrc=np.radians([30])
noise_para=1e-18
type='moving'
omega_src=1.5e-4*Fsamp

x=func.rx_signal(Rant,Tant,Rsrc,Tsrc,s,Fsrc,Fsamp=Fsamp,alpha=noise_para,type=type,omegaB=omega_src)

# Coarse beamforming step

Ncoarse=15**2
coarse_grid_size=5e-2
ds=2*coarse_grid_size/np.sqrt(Ncoarse)
spacetime=False
grid='cart'
omega_est=0

y=func.sum_signals(x,Ncoarse,coarse_grid_size,Rant,Tant,Fsamp=Fsamp,spacetime=spacetime,omega_est=omega_est)

Y,Yf=func.katydid_fft(y)
pow_Y=(abs(Y)**2)
f_max_ind=np.argmax(np.max(pow_Y,axis=(0,1)))
grid_edge=int(np.sqrt(Ncoarse))
#coarse_pixel=np.unravel_index(np.argmax(pow_Y[:,:,f_max_ind]),(grid_edge,grid_edge))

clean_y=func.fft_window(y,sig=1.5,Nwindow=32)

mean_y=np.mean(abs(clean_y)**2,axis=2)
coarse_pixel=np.unravel_index(np.argmax(mean_y),(grid_edge,grid_edge))

plt.figure()
#plt.imshow(pow_Y[:,:,f_max_ind].T,origin='lower')
plt.imshow(mean_y.T,origin='lower')
plt.plot(coarse_pixel[0],coarse_pixel[1],'r.')
plt.show()

# Fine follow-up step
Nfine=15**2
fine_grid_size=2e-2
spacetime=False
fine_grid='cart'
omega_est=0
fine_grid_center=ds*np.array([coarse_pixel[0]-(grid_edge-1)//2,coarse_pixel[1]-(grid_edge-1)//2])
print(fine_grid_center)

y_fine=func.sum_signals(x,Nfine,fine_grid_size,Rant,Tant,grid_center=fine_grid_center,spacetime=spacetime,omega_est=omega_est)

Y_fine,Yf=func.katydid_fft(y_fine)
pow_Y_fine=(abs(Y_fine)**2)
mean_y_fine=np.mean(abs(y_fine)**2,axis=2)
plt.figure()
#plt.imshow(pow_Y_fine[:,:,f_max_ind].T,origin='lower')
plt.imshow(mean_y_fine.T,origin='lower')
plt.show()


#cleaned_y=func.fft_window(y,sig=1.5,Nwindow=32)

#mean_y=np.mean(abs(cleaned_y)**2,axis=2).T

#max_y=np.max(mean_y)

