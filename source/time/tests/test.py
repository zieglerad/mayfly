import numpy as np
import matplotlib.pyplot as plt 
import time_domain.func as func 

array_radius=10e-02
array_sep=6

Rant,Tant=func.array_geo(array_radius,array_sep)


Flo=25e9
Fsrc=np.array([24.95e9])
Fsamp=500e6
Nsamp=256
As=np.array([1e-6])

s=func.source_signal(Fsrc,Flo,Fsamp,Nsamp,As)

Rsrc=[2e-2]
Tsrc=np.radians([0])
noise_para=0
type='moving'
omega_src=12e-3*Fsamp

x=func.rx_signal(Rant,Tant,Rsrc,Tsrc,s,Fsrc,alpha=noise_para,type=type,omegaB=omega_src)

#plt.figure()
#for n in range(60):
#    plt.plot(x[n,:100])

#plt.show()

Ngrid=101**2
physics_rad=5e-2
spacetime=True
grid='cart'
omega_est=0

y=func.sum_signals(x,Ngrid,physics_rad,Rant,Tant,spacetime=spacetime,omega_est=omega_est)
#print(y.shape)

#Y=np.fft.fft(y)

plt.figure()
if not spacetime:
    plt.imshow(np.mean(np.real(y)**2,axis=2).T,origin='lower')
    #plt.imshow((abs(Y)**2)[:,:,819].T,origin='lower')
else:
    plt.imshow(np.mean(np.real(y)**2,axis=2),origin='lower')
    #plt.imshow((abs(Y)**2)[:,:,819].T,origin='lower')
plt.show()