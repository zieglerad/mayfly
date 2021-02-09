import numpy as np
import matplotlib.pyplot as plt 
import time_domain.func as func 
from skimage import feature
from skimage import transform
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
Tsrc=np.radians([0])
noise_para=1e-16
type='moving'
omega_src=2e-4*Fsamp

x=func.rx_signal(Rant,Tant,Rsrc,Tsrc,s,Fsrc,alpha=noise_para,type=type,omegaB=omega_src)

#plt.figure()
#for n in range(60):
#    plt.plot(x[n,:100])

#plt.show()

Ngrid=41**2
physics_rad=5e-2
spacetime=False
grid='cart'
omega_est=0

y=func.sum_signals(x,Ngrid,physics_rad,Rant,Tant,spacetime=spacetime,omega_est=omega_est)

cleaned_y=func.fft_window(y,sig=1.5,Nwindow=32)

Y,Yf=func.katydid_fft(y)
f_max_ind=np.argmax(np.max(Y,axis=(0,1)))

mean_y=np.mean(abs(cleaned_y)**2,axis=2).T
max_y=np.max(mean_y)
edges=feature.canny(mean_y,sigma=1.0,low_threshold=max_y*0.1,
high_threshold=max_y*.5)
#print(transform.hough_ellipse(edges,min_size=1,accuracy=100,threshold=10))
if not spacetime:
#    plt.figure()
#    plt.imshow(np.mean(abs(y)**2,axis=2).T,origin='lower')
#    plt.title('y')
#    plt.figure()
#    plt.imshow(mean_y,origin='lower')
#    plt.title('cleaned y')
#    plt.figure()
#    plt.imshow((abs(Y)**2)[:,:,f_max_ind].T,origin='lower')
#    plt.title('FFT y')
    plt.figure()
    plt.imshow(edges,origin='lower',cmap='viridis')
    plt.title('edge image')
#else:
#    plt.imshow(np.mean(np.real(y)**2,axis=2),origin='lower')
    #plt.imshow((abs(Y)**2)[:,:,819].T,origin='lower')
plt.show()
