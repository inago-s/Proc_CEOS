import os
import struct
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr


def get_fileinfo(folder_name):
    filelist = os.listdir(folder_name)
    HH_file = HV_file = VV_file = VH_file = LED_file = nline = ncell = None

    for file in filelist:
        if 'IMG-HH' in file:
            HH_file = os.path.join(folder_name, file)
        if 'IMG-HV' in file:
            HV_file = os.path.join(folder_name, file)
        if 'IMG-VV' in file:
            VV_file = os.path.join(folder_name, file)
        if 'IMG-VH' in file:
            VH_file = os.path.join(folder_name, file)
        if 'LED' in file:
            LED_file = os.path.join(folder_name, file)

    if HH_file is not None:
        with open(HH_file, mode='rb') as f:
            f.seek(236)
            nline = int(f.read(8))
            f.seek(248)
            ncell = int(f.read(8))
    elif HV_file is not None:
        with open(HV_file, mode='rb') as f:
            f.seek(236)
            nline = int(f.read(8))
            f.seek(248)
            ncell = int(f.read(8))
    elif VV_file is not None:
        with open(VV_file, mode='rb') as f:
            f.seek(236)
            nline = int(f.read(8))
            f.seek(248)
            ncell = int(f.read(8))
    elif VH_file is not None:
        with open(VH_file, mode='rb') as f:
            f.seek(236)
            nline = int(f.read(8))
            f.seek(248)
            ncell = int(f.read(8))
    else:
        print('IMG file is not found.')
        exit()

    return HH_file, HV_file, VV_file, VH_file, nline, ncell, LED_file


def get_gcp(file, L, P):
    gcp = []
    with open(file, mode='rb') as f:
        gcp_L = np.linspace(0, L-1, int((L/1000)+1), dtype='int')

        for line in gcp_L:
            f.seek(720+(line)*(P*8+544)+192)
            f_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            f_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000

            gcp.append(gdal.GCP(f_lon, f_lat, 0.0, float(0), float(line)))
            gcp.append(gdal.GCP(m_lon, m_lat, 0.0, float((P/2)), float(line)))
            gcp.append(gdal.GCP(e_lon, e_lat, 0.0, float(P), float(line)))

    return gcp


def get_gcp2(file, L, P):
    with open(file, mode='rb') as f:
        f.seek(1604432+1024)
        # 緯度変換係数
        coefficient_lat = [float(f.read(20)) for i in range(25)]
        # 経度変換係数
        coefficient_lon = [float(f.read(20)) for i in range(25)]

        gcp_L = np.linspace(0, L, int((L/1000)+1), dtype='int')
        gcp_P = np.linspace(0, P, int((P/1000)+1), dtype='int')
        gcp = []

        for line in gcp_L:
            for pixel in gcp_P:
                l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
                p_matrix = np.array(
                    [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
                lp_matrix = np.ravel(l_matrix*p_matrix)
                gcp.append(gdal.GCP(np.dot(coefficient_lon, lp_matrix),
                                    np.dot(coefficient_lat, lp_matrix),
                                    0.0, float(pixel), float(line)))

        return gcp


def make_intensity(nlines, npixels, file, CF, name):
    sigma = np.empty((nlines, npixels))
    phase = np.empty((nlines, npixels))
    with open(file, mode='rb') as fp:
        nrec = 544+npixels*8
        for h in range(nlines):
            fp.seek(720+nrec*h)
            data = struct.unpack(">%s" % (int((nrec)/4)) +
                                 "f", fp.read(int(nrec)))
            data = np.array(data).reshape(-1, int(nrec/4))
            data = data[:, int(544/4):int(nrec/4)]
            slc = data[:, ::2] + 1j*data[:, 1::2]
            sigma[h] = 20*np.log10(abs(slc))+CF-32.0
            phase[h] = np.angle(slc)

    sigma = np.array(255*(sigma-np.amin(sigma)) /
                     (np.amax(sigma)-np.amin(sigma)), dtype="uint8")
    sigma = cv2.equalizeHist(sigma)
    sigma = cv2.medianBlur(sigma, 5)
    plt.imsave(name+'.jpg', sigma, cmap='gray')

    phase = np.array(255*(phase - np.amin(phase)) /
                     (np.amax(phase) - np.amin(phase)), dtype="uint8")
    plt.imsave(name+'_phase.jpg', phase, cmap='jet')


def main():
    if len(sys.argv) < 2:
        print("you need to select the folder")
        exit()
    else:
        folder_name = sys.argv[1]

    HH_file, HV_file, VV_file, VH_file, nlines, npixels, LED_file =\
        get_fileinfo(folder_name)

    with open(LED_file, mode='rb') as f:
        f.seek(740)
        seen = f.read(32).decode().strip()
        f.seek(25900)
        CF = float(f.read(16))

    file_name = os.path.join(folder_name, seen)

    make_intensity(nlines, npixels, HH_file, CF, file_name+'HH')
    gcpList = get_gcp(HH_file, nlines, npixels)
    # gcpList = get_gcp2(LED_file, nlines, npixels)

    with open(os.path.join(folder_name, seen)+'.points', mode='w') as f:
        f.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual\n")
        for g in gcpList:
            s = ",".join(
                [str(g.GCPX), str(g.GCPY), str(g.GCPPixel), str(-g.GCPLine), '1', '0', '0', '0\n'])
            f.writelines(s)


if __name__ == '__main__':
    main()
