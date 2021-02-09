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
Tsrc=np.radians([0])
noise_para=1e-14
type='moving'
omega_src=2e-4*Fsamp

x=func.rx_signal(Rant,Tant,Rsrc,Tsrc,s,Fsrc,alpha=noise_para,type=type,omegaB=omega_src)

#plt.figure()
#for n in range(60):
#    plt.plot(x[n,:100])

#plt.show()

Ngrid=41**2
physics_rad=5e-2
ds=physics_rad/np.sqrt(Ngrid)
spacetime=False
grid='cart'
omega_est=0

y=func.sum_signals(x,Ngrid,physics_rad,Rant,Tant,spacetime=spacetime,omega_est=omega_est)

cleaned_y=func.fft_window(y,sig=1.5,Nwindow=32)

Y,Yf=func.katydid_fft(y)
f_max_ind=np.argmax(np.max(Y,axis=(0,1)))

mean_y=np.mean(abs(cleaned_y)**2,axis=2).T

mean_1=np.mean(abs(cleaned_y[:,:,:Nsamp//3])**2,axis=2).T
mean_2=np.mean(abs(cleaned_y[:,:,2*Nsamp//3:])**2,axis=2).T


max_y=np.max(mean_y)
edges=feature.canny(mean_y,sigma=1.0,low_threshold=max_y*0.1,
high_threshold=max_y*.5)

#spot=morphology.area_closing(edges,connectivity=2)

#plt.figure()
#plt.imshow(spot,origin='lower')
#plt.show()

edge_ind=np.argwhere(edges>0)

edge_center=np.mean(edge_ind,axis=0)

#print(edge_ind-edge_center)
shape=edge_ind-edge_center
scaled_shape=shape*ds
print(scaled_shape)

#plt.figure()
#plt.plot(shape[:,0],shape[:,1],'b.')
#plt.show()

#squared_deviations=np.sum(shape**2,axis=1)
#sq_dev_sorted_ind=np.argsort(squared_deviations)
#major_1=shape[sq_dev_sorted_ind[-1],:]
#major_2=shape[sq_dev_sorted_ind[-2],:]
#major=np.array([major_1,major_2])

#print(major,major[:,0])

#major_reg=stats.linregress(major[:,0],major[:,1])

#x_r=np.arange(-2.5,2.5,0.1)
#y_r=np.arange(-1,1,.1)
#major_line=major_reg[0]*x_r+major_reg[1]
#minor_line=-1/major_reg[0]*y_r-major_reg[1]

#plt.figure()
#plt.plot(shape[:,0],shape[:,1],'b')
#plt.plot(x_r,major_line,'r')
#plt.plot(y_r,minor_line,'r')
#plt.show()

if not spacetime:
#    plt.figure()
#    plt.imshow(np.mean(abs(y)**2,axis=2).T,origin='lower')
#    plt.title('y')
    plt.figure()
    plt.imshow(mean_1,origin='lower')
    plt.title('y1')
    plt.figure()
    plt.imshow(mean_2,origin='lower')
    plt.title('y2')
    plt.figure()
    plt.imshow(edges,origin='lower',cmap='viridis')
    plt.plot(edge_center[1],edge_center[0],'r.')
    plt.title('edge image')
#else:
#    plt.imshow(np.mean(np.real(y)**2,axis=2),origin='lower')
    #plt.imshow((abs(Y)**2)[:,:,819].T,origin='lower')
plt.show()
