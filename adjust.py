from re import L
from PIL import Image
from matplotlib import pyplot as plt
from numpy.lib.shape_base import dstack
from CEOS_module.Proc_CEOS import Proc_CEOS
import itertools
from joblib import Parallel, delayed
import numpy as np
import struct
from scipy import interpolate
import math
import os

DEM_PATH = '../DEM'
DEM_LIST = {}


def get_DEM(lon, lat):
    filepath = os.path.join(DEM_PATH, 'ALPSMLC30_N' +
                            str(int(lat)).zfill(3)+'E'+str(int(lon)).zfill(3)+'_DSM.tif')

    if filepath not in DEM_LIST.keys():
        img = Image.open(filepath)
        DEM_LIST[filepath] = np.array(img)

    h_l, w_l = DEM_LIST[filepath].shape

    h, w = (h_l-1)-int((lat-int(lat))/(1 / h_l)
                       ), int((lon-int(lon))/(1 / w_l))

    return DEM_LIST[filepath][h][w]


def get_sat_pos(led, img, y, h):
    img.seek(720+44)
    time_start = struct.unpack(">i", img.read(4))[0]/1000
    led.seek(720+500)

    led.seek(720+4096+140)
    position_num = int(led.read(4))
    led.seek(720+4096+182)
    time_interval = int(float(led.read(22)))

    led.seek(720+4096+160)
    start_time = float(led.read(22))

    led.seek(720+68)
    center_time = led.read(32)
    Hr = float(center_time[8:10])*3600
    Min = float(center_time[10:12])*60
    Sec = float(center_time[12:14])
    msec = float(center_time[14:17])*1e-3
    center_time = Hr+Min+Sec+msec
    time_end = time_start + (center_time - time_start)*2

    img.seek(236)
    nline = int(img.read(8))
    time_obs = np.arange(time_start, time_end, (time_end - time_start)/nline)
    time_pos = np.arange(start_time, start_time +
                         time_interval*position_num, time_interval)
    pos_ary = []

    led.seek(720+4096+386)
    for _ in range(position_num):
        for _ in range(3):
            pos = float(led.read(22))
            pos_ary.append(pos)
        led.read(66)
    pos_ary = np.array(pos_ary).reshape(-1, 3)

    fx = interpolate.interp1d(time_pos, pos_ary[:, 0], kind="cubic")
    fy = interpolate.interp1d(time_pos, pos_ary[:, 1], kind="cubic")
    fz = interpolate.interp1d(time_pos, pos_ary[:, 2], kind="cubic")
    X = fx(time_obs)
    Y = fy(time_obs)
    Z = fz(time_obs)
    XYZ = dstack([X, Y, Z])

    return XYZ[0][y:y+h, :]


def lla2ecef(lat, lon, alt):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 1 - (1 - f) * (1 - f)
    v = a / math.sqrt(1 - e2 * math.sin(lat) * math.sin(lat))

    x = (v + alt) * math.cos(lat) * math.cos(lon)
    y = (v + alt) * math.cos(lat) * math.sin(lon)
    z = (v * (1 - e2) + alt) * math.sin(lat)
    return np.array([x, y, z])


def main():
    C = Proc_CEOS('../ALOS2014410740-140829')
    img = open(C.HH_file, mode='rb')
    led = open(C.LED_file, mode='rb')

    x, y = 0, 0
    h, w = 1024, 1024

    C.save_intensity_img(C.HH_file, x, y, w, h, '.', 'slc.png')

    orbit = get_sat_pos(led, img, y, h)

    lonlat = Parallel(n_jobs=-1)(delayed(C.get_coordinate)
                                 (s_x, s_y)for s_x, s_y in itertools.product(range(x, x+w), range(y, y+h)))
    lonlat = np.array(lonlat)
    dem = Parallel(n_jobs=-1, require='sharedmem')(delayed(get_DEM)
                                                   (ll[0], ll[1])for ll in lonlat)

    C.save_DEM_img(dem, x, y, w, h, '.', 'hoge.png')

    # sl = np.zeros(np.shape(h, w))
    # for i in range(h):
    #     for j in range(w):
    #         ixyz = lla2ecef(lonlat[i*h+j][1], lonlat[i*h+j][0], dem[i*h+j]+40)
    #         sl[i, j] = np.linalg.norm(orbit[i]-ixyz)


if __name__ == "__main__":
    main()
