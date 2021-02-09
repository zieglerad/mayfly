# This script parses egg files in the specified directory.
# Takes the FFT and extracts the main peaks in the spectrum.

## import ##
import MFParse
import MFTime
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
#####

## global ##
array_rad=0.1
Vrange=(80-280*array_rad+300*array_rad**2)*1e-9
sliceSize=8192
nSlice=2
peakThreshold=0.1
badEgg=[]
eggPaths=[]
frequencyPeakDicts=[]
###

## script command line ##
parser=argparse.ArgumentParser()
parser.add_argument('targetPath',type=str)
parser.add_argument('destPath',type=str)

args=parser.parse_args()
####

## functions ##
def getEggFilePaths(dirPath,listOfPaths):
    try:
        os.path.isdir(dirPath)
    except:
        return False

    for fPath in os.listdir(dirPath):
        if fPath.split('.')[-1]=='egg':
            listOfPaths.append(os.path.join(dirPath,fPath))

    return True

def getEggFileParams(eggPath,params):
    try:
        os.path.isfile(eggPath)
    except:
        return False

    try:
        pitchAngle=float(eggPath.split('/')[-1].split('Angle')[-1].split('_')[0])
        radius=float(eggPath.split('/')[-1].split('Pos')[-1].split('.egg')[0])
    except:
        return False

    params.append(pitchAngle)
    params.append(radius)

    return True

####

if getEggFilePaths(args.targetPath,eggPaths):

    for eggPath in eggPaths:

        simulationParams=[]
        getEggFileParams(eggPath,simulationParams)

        eggDataTime=MFParse.parse_egg(eggPath,Vrange=Vrange)
        eggSliceTime=MFParse.slice_egg(eggDataTime,nSlice,sliceSize)

        if MFParse.check_egg_slice(eggSliceTime):
            badEgg.append(eggPath)
            print(eggPath)
        else:
            eggSliceFFT,FFTFrequencies=MFTime.katydid_fft(eggSliceTime)
            frequencyPeaks=np.where(abs(eggSliceFFT[0,:])>peakThreshold*np.max(abs(eggSliceFFT[0,:])))[0]
            frequencyPeakAmplitudes=abs(eggSliceFFT[0,frequencyPeaks])
            frequencyPeakDicts.append({'param':simulationParams,'inds':frequencyPeaks,'amps':frequencyPeakAmplitudes})
    with open(args.destPath,'wb') as outfile:
        pkl.dump(frequencyPeakDicts,outfile)
