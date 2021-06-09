import itertools
import os
import struct
from typing import NoReturn, Tuple
import numpy as np
import matplotlib.pyplot as plt


class Proc_CEOS:
    def __init__(self, folder) -> None:
        self.folder = folder
        self.HH_file = self.HV_file = self.VV_file = self.VH_file \
            = self.LED_file = self.nline = self.ncell = None

        filelist = os.listdir(folder)

        for file in filelist:
            if 'IMG-HH' in file:
                self.HH_file = os.path.join(folder, file)
            if 'IMG-HV' in file:
                self.HV_file = os.path.join(folder, file)
            if 'IMG-VV' in file:
                self.VV_file = os.path.join(folder, file)
            if 'IMG-VH' in file:
                self.VH_file = os.path.join(folder, file)
            if 'LED' in file:
                self.LED_file = os.path.join(folder, file)

        if self.HH_file is not None:
            with open(self.HH_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.main_file = self.HH_file
        elif self.HV_file is not None:
            with open(self.HV_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.main_file = self.HV_file
        elif self.VV_file is not None:
            with open(self.VV_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.main_file = self.VV_file
        elif self.VH_file is not None:
            with open(self.VH_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.main_file = self.VH_file
        else:
            print('IMG file connot be found.')
            exit()

        if self.LED_file is not None:
            with open(self.LED_file, mode='rb') as f:
                f.seek(740)
                self.seen_id = f.read(32).decode().strip()
                f.seek(25900)
                self.CF = float(f.read(16))
                f.seek(1604432+1024)
                self.coefficient_lat = [float(f.read(20)) for _ in range(25)]
                self.coefficient_lon = [float(f.read(20)) for _ in range(25)]
        else:
            print('LED file connot be found.')
            exit()

    def get_gcp_three(self) -> list:
        gcp = []
        with open(str(self.main_file), mode='rb') as f:
            gcp_L = np.linspace(
                0, self.nline-1, int((self.nline/1000)+1), dtype='int')
            print(gcp_L)
            for line in gcp_L:
                f.seek(int(720+(line)*(self.ncell*8+544)+192))
                f_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
                m_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
                e_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
                f_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
                m_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
                e_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000

                gcp.append([f_lon, f_lat, 0.0, float(0), float(line)])
                gcp.append([m_lon, m_lat, 0.0, float(0), float(line)])
                gcp.append([e_lon, e_lat, 0.0, float(0), float(line)])

        return gcp

    def get_lon_lat(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        return np.dot(self.coefficient_lon, lp_matrix), np.dot(self.coefficient_lat, lp_matrix)

    def make_gcp(self, x, y, w, h, folder) -> None:
        with open(os.path.join(folder, self.seen_id)+str(y)+'-'+str(x)+'.points', mode='w') as f:
            s = ""
            for _x, _y in itertools.product(np.linspace(x, x+w, 5).astype('int'), np.linspace(y, y+h, 5).astype('int')):
                s += " -gcp "
                lon, lat = self.get_lon_lat(x+_x, y+_y)
                s += " ".join(
                    [str(_x), str(_y), str(lon), str(lat)])
            f.write(s)

    def make_intensity_fig(self, Pol_file, x=0, y=0, w=None, h=None, folder=None) -> Tuple[np.ndarray, np.ndarray]:
        if x < 0 or x > self.ncell:
            print('input error')
            exit()

        if y < 0 or y > self.nline:
            print('input error')
            exit()

        if w is None:
            w = self.ncell-x
        elif w > self.ncell-x:
            print('input error')
            exit()

        if h is None:
            h = self.nline-y
        elif h > self.nline-y:
            print('input error')
            exit()

        sigma, phase = self.get_intensity(Pol_file, x, y, w, h)
        sigma_img = np.array(255*(sigma-np.amin(sigma)) /
                             (np.amax(sigma)-np.amin(sigma)), dtype="uint8")
        phase_img = np.array(255*(phase - np.amin(phase)) /
                             (np.amax(phase) - np.amin(phase)), dtype="uint8")
        name = str(self.seen_id)+'_' + \
            Pol_file.split('-')[2]+'_'+str(y)+'-'+str(x)
        if folder is None:
            plt.imsave(os.path.join(self.folder, name) +
                       '_sigma.png', sigma_img, cmap='gray')
            plt.imsave(os.path.join(self.folder, name) +
                       '_phase.png', phase_img, cmap='jet')
        else:
            plt.imsave(os.path.join(folder, name) +
                       '_sigma.png', sigma, cmap='gray')
            plt.imsave(os.path.join(folder, name) +
                       '_phase.png', phase, cmap='jet')

        return sigma, phase

    def get_intensity(self, Pol_file, x=0, y=0, w=None, h=None) -> Tuple[np.ndarray, np.ndarray]:
        if x < 0 or x > self.ncell:
            print('input error')
            exit()

        if y < 0 or y > self.nline:
            print('input error')
            exit()

        if w is None:
            w = self.ncell-x
        elif w > self.ncell-x:
            print('input error')
            exit()

        if h is None:
            h = self.nline-y
        elif h > self.nline-y:
            print('input error')
            exit()

        nrec = 544+self.ncell*8

        with open(Pol_file, mode='rb') as fp:
            fp.seek(720+int(nrec*y))
            data = struct.unpack(
                ">%s" % (int((nrec*h)/4))
                + "f", fp.read(int(nrec*h)))
            data = np.array(data).reshape(-1, int(nrec/4))
            data = data[:, int(544/4):int(nrec/4)]
            slc = data[:, ::2] + 1j*data[:, 1::2]
            slc = slc[:, x:x+w]

        sigma = 20*np.log10(abs(slc))+self.CF-32.0
        phase = np.angle(slc)

        return sigma, phase
