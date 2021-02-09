__name__='MFTime'


import numpy as np
import scipy.signal as signal
import scipy.constants as const
import matplotlib.pyplot as plt

def array_geo(radius,sep,shape='cir'):
    if shape=='cir':
        r_ant=radius*np.ones(360//sep)
        theta_ant=np.radians(np.arange(0,360,sep))
    return r_ant, theta_ant

def source_signal(Fsrc,Flo,Fsamp,Nsamp,As):

    time=np.arange(0,Nsamp,1)*(1/Fsamp)
    As=np.expand_dims(As,axis=1)
    As=np.repeat(As,time.size,axis=1)
    s_src=As*np.exp(-1j*2*np.pi*np.outer((Fsrc-Flo),time))

    return s_src

def rx_signal(r_ant,theta_ant,rsrc,thetasrc,s_src,Fsrc,Fsamp=500e6,alpha=1e-18,type='stationary',omegaB=0):

    wlsrc=3e8/Fsrc

    if type=='stationary':
        rsrc=np.expand_dims(rsrc,axis=1)
        thetasrc=np.expand_dims(thetasrc,axis=1)
        rsrc=np.repeat(rsrc,r_ant.size,axis=1)
        thetasrc=np.repeat(thetasrc,theta_ant.size,axis=1)

        r_ant=np.expand_dims(r_ant,axis=(0))
        theta_ant=np.expand_dims(theta_ant,axis=(0))

        r_ant=np.repeat(r_ant,rsrc.shape[0],axis=0)
        theta_ant=np.repeat(theta_ant,thetasrc.shape[0],axis=0)

    elif type=='moving':

        wlsrc=np.expand_dims(wlsrc,axis=(1,2))
        wlsrc=np.repeat(wlsrc,s_src.size,axis=2)
        wlsrc=np.repeat(wlsrc,r_ant.size,axis=1)

        rsrc=np.expand_dims(rsrc,axis=(1,2))
        rsrc=np.repeat(rsrc,s_src.size,axis=2)
        rsrc=np.repeat(rsrc,r_ant.size,axis=1)

        thetasrc=np.expand_dims(thetasrc,axis=(1,2))
        thetasrc=np.repeat(thetasrc,s_src.size,axis=2)
        omegatime=omegaB*np.arange(0,s_src.size,1)*(1/Fsamp)
        thetasrc+=omegatime
        thetasrc=np.repeat(thetasrc,theta_ant.size,axis=1)

        r_ant=np.expand_dims(r_ant,axis=(0,2))
        theta_ant=np.expand_dims(theta_ant,axis=(0,2))

        r_ant=np.repeat(r_ant,s_src.size,axis=2)
        theta_ant=np.repeat(theta_ant,s_src.size,axis=2)
        r_ant=np.repeat(r_ant,rsrc.shape[0],axis=0)
        theta_ant=np.repeat(theta_ant,thetasrc.shape[0],axis=0)


    z=np.sqrt(rsrc**2+r_ant**2-2*rsrc*r_ant*np.cos(theta_ant-thetasrc))
    a=(1/z)*np.exp(-1j*2*np.pi*z/wlsrc)
    #a=np.exp(-1j*2*np.pi*z/wlsrc)
    noise_var=alpha

    wgn=np.random.multivariate_normal(np.zeros(2),0.5*np.eye(2)*np.sqrt(noise_var),size=(r_ant.shape[1],s_src.shape[1])).view(np.complex128)
    wgn=wgn.reshape(r_ant.shape[1],s_src.shape[1])

    if type=='stationary':
        x=np.matmul(a.T,s_src)+wgn
    elif type=='moving':
        for n in range(a.shape[0]):
            for m in range(a.shape[1]):
                a[n,m,:]*=s_src[n]
        a=np.sum(a,axis=0)

        x=a+wgn

    return x

def sum_signals(x,Ngrid,grid_size,Rant,Tant,grid='cart',Fest=25e9,Fsamp=500e6,spacetime=False,omega_est=0,use_antispiral=False,grid_center=np.array([0,0])):
    Nsamp=x.shape[1]
    Nch=x.shape[0]
    wl=3e8/Fest
    time=np.arange(0,Nsamp,1)*(1/Fsamp)
    shifted_channel_max_freqs=np.zeros((Ngrid,Nch))
    if grid =='cart':
        ind=np.arange(-(np.sqrt(Ngrid)+1)//2+1,(np.sqrt(Ngrid)+1)//2,1)/np.sqrt(Ngrid)
        xgrid=2*grid_size*ind+grid_center[0]
        ygrid=2*grid_size*ind+grid_center[1]
        #print(xgrid,ygrid)
        xx,yy=np.meshgrid(xgrid,ygrid)
        R0=np.sqrt(xx**2+yy**2)
        T0=np.arctan2(yy,xx)
    elif grid=='pol':#need to add the grid center offset to work here.
        ind=np.arange(0,int(np.sqrt(Ngrid)),1)/np.sqrt(Ngrid)
        R0=grid_size*ind
        T0=2*np.pi*ind
        R0,T0=np.meshgrid(R0,T0)
    elif grid=='fib':
        ind=np.arange(0,Ngrid,1)
        ep=1/2
        R0=grid_size*((ind+ep)/(Ngrid-1+2*ep))**0.5
        #print(grid_size)
        T0=2*np.pi*ind*const.golden
        R_offset=np.sqrt((grid_center**2).sum())
        T_offset=np.arctan2(grid_center[-1],grid_center[0])
        #print(R0)
        R0+=R_offset
        T0+=T_offset

    if spacetime and grid in ['cart','pol']:
        R0=np.expand_dims(R0,axis=-1)
        R0=np.expand_dims(R0,axis=-1)
        R0=np.repeat(R0,Nch,axis=-2)
        R0=np.repeat(R0,Nsamp,axis=-1)

        T0=np.expand_dims(T0,axis=-1)
        T0=np.expand_dims(T0,axis=-1)
        T0=np.repeat(T0,Nch,axis=-2)
        T0=np.repeat(T0,Nsamp,axis=-1)

        omegaT=time*omega_est
        omegaT=np.expand_dims(omegaT,axis=(0))
        omegaT=np.expand_dims(omegaT,axis=(0))
        omegaT=np.expand_dims(omegaT,axis=(0))
        omegaT=np.repeat(omegaT,int(np.sqrt(Ngrid)),axis=0)
        omegaT=np.repeat(omegaT,int(np.sqrt(Ngrid)),axis=1)
        omegaT=np.repeat(omegaT,Nch,axis=2)

        R=R0
        T=T0+omegaT

        Rant=np.expand_dims(Rant,axis=1)
        Rant=np.repeat(Rant,Nsamp,axis=1)

        Tant=np.expand_dims(Tant,axis=1)
        Tant=np.repeat(Tant,Nsamp,axis=1)
        #print(Rant.shape,Tant.shape)

        y=np.zeros((int(np.sqrt(Ngrid)),int(np.sqrt(Ngrid)),Nsamp),
        dtype=np.complex128)
        for i in range(int(np.sqrt(Ngrid))):
            for j in range(int(np.sqrt(Ngrid))):
                z=np.sqrt(R[i,j]**2+Rant**2-2*R[i,j]*Rant*np.cos(T[i,j]-Tant))
                a=(np.exp(-1j*2*np.pi*z/wl))
                #a=(1/z)*np.exp(-1j*2*np.pi*z/wl)
                if use_antispiral:
                    antispiral_phases=np.exp(1j*2*np.pi*np.arange(0,Nch,1)/Nch)
                    antispiral_phases=np.expand_dims(antispiral_phases,axis=(1))

                    antispiral_phases=np.repeat(antispiral_phases,Nsamp,axis=1)
                    a*=antispiral_phases
                y[i,j]=(a.conjugate()*x).sum(0)

    elif spacetime and grid in ['fib']:
        R0=np.expand_dims(R0,axis=-1)
        R0=np.expand_dims(R0,axis=-1)
        R0=np.repeat(R0,Nch,axis=-2)
        R0=np.repeat(R0,Nsamp,axis=-1)

        T0=np.expand_dims(T0,axis=-1)
        T0=np.expand_dims(T0,axis=-1)
        T0=np.repeat(T0,Nch,axis=-2)
        T0=np.repeat(T0,Nsamp,axis=-1)

        omegaT=time*omega_est
        omegaT=np.expand_dims(omegaT,axis=(0))
        omegaT=np.expand_dims(omegaT,axis=(0))
        omegaT=np.repeat(omegaT,Ngrid,axis=0)
        omegaT=np.repeat(omegaT,Nch,axis=1)

        R=R0
        T=T0+omegaT

        Rant=np.expand_dims(Rant,axis=1)
        Rant=np.repeat(Rant,Nsamp,axis=1)

        Tant=np.expand_dims(Tant,axis=1)
        Tant=np.repeat(Tant,Nsamp,axis=1)
        #print(Rant.shape,Tant.shape)

        y=np.zeros((Ngrid,Nsamp),dtype=np.complex128)
        for i in range(Ngrid):
            z=np.sqrt(R[i]**2+Rant**2-2*R[i]*Rant*np.cos(T[i]-Tant))
            a=np.exp(-1j*2*np.pi*z/wl)
            #a=(1/z)*np.exp(-1j*2*np.pi*z/wl)

            if use_antispiral:
                antispiral_phases=np.exp(1j*2*np.pi*np.arange(0,Nch,1)/Nch)
                antispiral_phases=np.expand_dims(antispiral_phases,axis=(1))

                antispiral_phases=np.repeat(antispiral_phases,Nsamp,axis=1)
                a*=antispiral_phases
            y_ch=a.conjugate()*x
            y_ch_fft=np.fft.fftshift(np.fft.fft(y_ch,axis=-1),axes=1)
            shifted_channel_max_freqs[i]=np.argmax(abs(y_ch_fft)**2,axis=-1)
            y[i]=(a.conjugate()*x).sum(0)
        #print(y.shape)
    elif not spacetime:
        if grid in ['cart','pol']:
            R=R0
            T=T0
            R=np.expand_dims(R,axis=0)
            T=np.expand_dims(T,axis=0)
            #print(T)
            R=np.repeat(R,Rant.size,axis=0)
            T=np.repeat(T,Rant.size,axis=0)
            Rant=np.expand_dims(Rant,axis=(1,2))
            Rant=np.repeat(Rant,R.shape[1],axis=(1))
            Rant=np.repeat(Rant,R.shape[1],axis=(2))
            Tant=np.expand_dims(Tant,axis=(1,2))
            Tant=np.repeat(Tant,T.shape[1],axis=(1))
            Tant=np.repeat(Tant,T.shape[1],axis=(2))

            z=np.sqrt(R**2+Rant**2-2*Rant*R*np.cos(T-Tant))
            a=np.exp(-1j*2*np.pi*z/wl)
            #a=(1/z)*np.exp(-1j*2*np.pi*z/wl)

            #a=np.exp(-1j*2*np.pi*z/wl)
            if use_antispiral:
                antispiral_phases=np.exp(1j*2*np.pi*np.arange(0,Nch,1)/Nch)
                antispiral_phases=np.expand_dims(antispiral_phases,axis=(1,2))

                antispiral_phases=np.repeat(antispiral_phases,int(np.sqrt(Ngrid)),axis=1)
                antispiral_phases=np.repeat(antispiral_phases,int(np.sqrt(Ngrid)),axis=2)
                a*=antispiral_phases

        elif grid=='fib':
            R=R0
            T=T0
            R=np.expand_dims(R,axis=0)
            R=np.repeat(R,Rant.size,axis=0)
            T=np.expand_dims(T,axis=0)
            T=np.repeat(T,Tant.size,axis=0)

            Rant=np.expand_dims(Rant,axis=1)
            Rant=np.repeat(Rant,R.shape[1],axis=1)
            Tant=np.expand_dims(Tant,axis=1)
            Tant=np.repeat(Tant,T.shape[1],axis=1)

            z=np.sqrt(R**2+Rant**2-2*Rant*R*np.cos(T-Tant))
            #a=Rant[0,0]*(1/z)*np.exp(-1j*2*np.pi*z/wl)
            a=np.exp(-1j*2*np.pi*z/wl)

            #a=np.exp(-1j*2*np.pi*z/wl)
            if use_antispiral:
                antispiral_phases=np.exp(1j*2*np.pi*np.arange(0,Nch,1)/Nch)
                antispiral_phases=np.expand_dims(antispiral_phases,axis=(1))
                antispiral_phases=np.repeat(antispiral_phases,Ngrid,axis=1)
                a*=antispiral_phases
        y=np.dot(a.conjugate().T,x)

    return y,shifted_channel_max_freqs,R,T

def katydid_fft(y,fsamp=200e6):

    Nsamp=y.shape[-1]
    Ngrid=y.shape[0]
    if y.ndim==3:
        fft=np.zeros((Ngrid,Ngrid,Nsamp),dtype=np.complex128)
        #print(fft.shape,y.shape)
        for i in range(Ngrid):
            for j in range(Ngrid):
                fft[i,j,:]=np.fft.fftshift(np.fft.fft(y[i,j,:]))
    elif y.ndim==2:
        fft=np.zeros((Ngrid,Nsamp),dtype=np.complex128)
        for i in range(Ngrid):
            fft[i,:]=np.fft.fftshift(np.fft.fft(y[i,:]))

    freqs=np.round(np.fft.fftshift(np.fft.fftfreq(Nsamp, 1/fsamp))+fsamp/2,2)
    return fft,freqs

def katydid_ifft(Y,fsamp=200e6):

    Nsamp=Y.shape[-1]
    Ngrid=Y.shape[0]
    #if y.ndim==3:
    #    fft=np.zeros((Ngrid,Ngrid,Nsamp),dtype=np.complex128)
    #    #print(fft.shape,y.shape)
    #    for i in range(Ngrid):
    #        for j in range(Ngrid):
    #            fft[i,j,:]=np.fft.fftshift(np.fft.fft(y[i,j,:]))
    if Y.ndim==2:
        ifft=np.zeros((Ngrid,Nsamp),dtype=np.complex128)
        for i in range(Ngrid):
            ifft[i,:]=np.fft.ifft(np.fft.ifftshift(Y[i,:]))

    #freqs=np.round(np.fft.fftshift(np.fft.fftfreq(Nsamp, 1/fsamp))+fsamp/2,2)
    return ifft

def fft_window(y,type='summed',window='gaussian',sig=1.2,Nwindow=16,fsamp=200e6,win_ind=-666):
    Nsamp=y.shape[-1]
    Ngrid=y.shape[0]

    y_fft=np.fft.fftshift(np.fft.fft(y),axes=-1)
    if win_ind==-666:
        f_ind=np.argmax(np.max(y_fft,axis=(0,1)))
    else:
        f_ind=win_ind
    #print(f_ind)
    f=np.fft.fftshift(np.fft.fftfreq(Nsamp,1/fsamp),axes=-1)

    #f_ind=np.argmin(abs(f-win_freq))
    front_pad_size=f_ind-(Nwindow-1)//2
    back_pad_size=Nsamp-f_ind-(Nwindow-1)//2-1

    window_array=signal.windows.get_window((window,sig),Nwindow)[1:]
    padded_window=np.concatenate((np.zeros(front_pad_size),
    window_array,np.zeros(back_pad_size)))

    windowed_fft=y_fft
    if type=='summed':
        for i in range(windowed_fft.shape[0]):
            for j in range(windowed_fft.shape[1]):
                windowed_fft[i,j,:]*=padded_window
    elif type=='raw':
        for i in range(windowed_fft.shape[0]):
            windowed_fft[i,:]*=padded_window

    filtered_y=np.fft.ifft(np.fft.ifftshift(windowed_fft,axes=-1))

    return filtered_y
