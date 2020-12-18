import struct
import sys
import os

import numpy as np
from osgeo import gdal
from osgeo import osr


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


def get_gcp(LED, L, P):
    with open(LED, mode='rb') as f:
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


def read_bin(file, h, nrec):
    with open(file, mode='rb') as fp:
        fp.seek(720+nrec*h)
        data = struct.unpack(">%s" % (int((nrec)/4))+"f", fp.read(int(nrec)))
        data = np.array(data).reshape(-1, int(nrec/4))
        data = data[:, int(544/4):int(nrec/4)]
        slc = data[:, ::2] + 1j*data[:, 1::2]

    return slc


def Process_Pauli(pic, nlines, npixels, HH, HV, VV, VH, CF):
    nrec = 544+npixels*8
    for h in range(nlines):
        slc_hh, slc_hv, slc_vv, slc_vh = read_bin(HH, h, nrec),\
            read_bin(HV, h, nrec), read_bin(VV, h, nrec),\
            read_bin(VH, h, nrec)

        # r, g, b = abs(slc_hh-slc_hv), abs(slc_hv+slc_vh), abs(slc_hh+slc_vv)
        slc_hh = 20*log10(abs(slc_hh)) + CF - 32.0
        slc_hv = 20*log10(abs(slc_hh)) + CF - 32.0
        slc_vv = 20*log10(abs(slc_hh)) + CF - 32.0
        slc_vh = 20*log10(abs(slc_hh)) + CF - 32.0

        r, g, b = slc_hh-slc_hv, slc_hv+slc_vh, slc_hh+slc_vv
        pic[h] = np.stack([r, g, b], 2)


def Process_SLC(pic, nlines, npixels, HH, HV, VV, VH, CF):
    nrec = 544+npixels*8
    for h in range(nlines):
        slc_hh, slc_hv, slc_vv, slc_vh = read_bin(HH, h, nrec),\
            read_bin(HV, h, nrec), read_bin(VV, h, nrec),\
            read_bin(VH, h, nrec)

        slc_hh = 20*log10(abs(slc_hh)) + CF - 32.0
        slc_hv = 20*log10(abs(slc_hh)) + CF - 32.0
        slc_vv = 20*log10(abs(slc_hh)) + CF - 32.0
        slc_vh = 20*log10(abs(slc_hh)) + CF - 32.0


def make_Tiff(img, gcpList, band, folder_name):
    dtype = gdal.GDT_Byte
    output = gdal.GetDriverByName('GTiff').Create(
        folder_name+'test.tif', img.shape[1], img.shape[0],
        band, dtype, options=['PHOTOMETRIC=RGB'])
    gcp_srs = osr.SpatialReference()
    gcp_srs.ImportFromEPSG(3717)
    gcp_crs_wkt = gcp_srs.ExportToWkt()
    output.SetGCPs(gcpList, gcp_crs_wkt)

    output.GetRasterBand(1).WriteArray(img[:, :, 0])
    output.GetRasterBand(2).WriteArray(img[:, :, 1])
    output.GetRasterBand(3).WriteArray(img[:, :, 2])

    output.FlushCache()
    output = None


def main():
    if len(sys.argv) < 2:
        print("you need to enter the sar folder")
        exit()
    else:
        folder_name = sys.argv[1]

    HH_file, HV_file, VV_file, VH_file, nlines, npixels, LED_file = \
        get_fileinfo(folder_name)

    with open(LED_file, mode='rb') as f:
        f.seek(25900)
        CF = float(f.read(16))

    pic = np.empty((nlines, npixels, 3), dtype=np.float32)

    Process_Pauli(pic, nlines, npixels, HH_file, HV_file, VV_file, VH_file, CF)

    max_r = np.percentile(pic[:, :, 0], q=95)
    max_g = np.percentile(pic[:, :, 1], q=95)
    max_b = np.percentile(pic[:, :, 2], q=95)

    pic[:, :, 0] = np.clip((pic[:, :, 0]/max_r)*255, 0, 255).astype('uint8')
    pic[:, :, 1] = np.clip((pic[:, :, 1]/max_g)*255, 0, 255).astype('uint8')
    pic[:, :, 2] = np.clip((pic[:, :, 2]/max_b)*255, 0, 255).astype('uint8')

    gcpList = get_gcp(LED_file, nlines, npixels)

    make_Tiff(pic, gcpList, 3, folder_name)


if __name__ == '__main__':
    main()
