import os
import re
import sys
import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def get_fileinfo(folder_name):
    with open(folder_name+'/summary.txt') as f:
        summary=f.read().split('\n')
    for sum in summary:
        if 'IMG-HH' in sum:
            sum=re.split(r'IMG-HH*',sum)
            HH_filename=folder_name+'/IMG-HH'+sum[1].replace('"','')
        if 'IMG-HV' in sum:
            sum=re.split(r'IMG-HV*',sum)
            HV_filename=folder_name+'/IMG-HV'+sum[1].replace('"','')
        if 'IMG-VV' in sum:
            sum=re.split(r'IMG-VV*',sum)
            VV_filename=folder_name+'/IMG-VV'+sum[1].replace('"','')
        if 'IMG-VH' in sum:
            sum=re.split(r'IMG-VH*',sum)
            VH_filename=folder_name+'/IMG-VH'+sum[1].replace('"','')
        if 'Pdi_NoOfPixels_0=' in sum:
            sum=re.split(r'Pdi_NoOfPixels_0=',sum)
            npixels=int(sum[1].replace('"',''))
        if 'Pdi_NoOfLines_0=' in sum:
            sum=re.split(r'Pdi_NoOfLines_0=',sum)
            nlines=int(sum[1].replace('"',''))
        if 'Img_SceneStartDateTime' in sum:
            sum=re.split(r'Img_SceneStartDateTime=',sum)
            time=sum[1][:-5].replace('"','').replace(':','-').replace(' ','_')
    
    return HH_filename,HV_filename,VV_filename,VH_filename,time


def main():    
    folder_name=sys.argv[1]
    HH_filename,HV_filename,VV_filename,VH_filename,time,npixels,nlines=get_fileinfo(folder_name)
    nrec=544+npixels*8
    pic=np.empty((nlines,npixels,3),dtype=np.float32)

    for h in range(nlines):
        with open(HH_filename,mode='rb') as fp_hh:
            fp_hh.seek(720+nrec*h)
            data = struct.unpack(">%s"%(int((nrec)/4))+"f",fp_hh.read(int(nrec)))
            data = np.array(data).reshape(-1,int(nrec/4))
            data = data[:,int(544/4):int(nrec/4)]
            slc_hh = data[:,::2] + 1j*data[:,1::2]

        with open(HV_filename,mode='rb') as fp_hv:
            fp_hv.seek(720+nrec*h)
            data = struct.unpack(">%s"%(int((nrec)/4))+"f",fp_hv.read(int(nrec)))
            data = np.array(data).reshape(-1,int(nrec/4))
            data = data[:,int(544/4):int(nrec/4)]
            slc_hv = data[:,::2] + 1j*data[:,1::2]
    

        with open(VV_filename,mode='rb') as fp_vv:
            fp_vv.seek(720+nrec*h)
            data = struct.unpack(">%s"%(int((nrec)/4))+"f",fp_vv.read(int(nrec)))
            data = np.array(data).reshape(-1,int(nrec/4))
            data = data[:,int(544/4):int(nrec/4)]
            slc_vv = data[:,::2] + 1j*data[:,1::2]

        with open(VH_filename,mode='rb') as fp_vh:
            fp_vh.seek(720+nrec*h)
            data = struct.unpack(">%s"%(int((nrec)/4))+"f",fp_vh.read(int(nrec)))
            data = np.array(data).reshape(-1,int(nrec/4))
            data = data[:,int(544/4):int(nrec/4)]
            slc_vh = data[:,::2] + 1j*data[:,1::2]

        r=abs(slc_hh-slc_vv)
        b=abs(slc_hh+slc_vv)
        g=abs(slc_hv+slc_vh)

        pic[h]=np.stack([r,g,b],2)

    max_r=np.percentile(pic[:,:,0],q=95)
    print(max_r)
    max_g=np.percentile(pic[:,:,1],q=95)
    print(max_g)
    max_b=np.percentile(pic[:,:,2],q=95)
    print(max_b)

    pic[:,:,0]=np.clip(pic[:,:,0]/max_r,0,1)
    pic[:,:,1]=np.clip(pic[:,:,1]/max_g,0,1)
    pic[:,:,2]=np.clip(pic[:,:,2]/max_b,0,1)
    
    #date=datetime.datetime.now()
    file_name=folder_name+'_'+time+'.png'
    plt.imsave(file_name,pic,)



if __name__ == '__main__':
    main()
