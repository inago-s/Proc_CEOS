import os
import struct
from typing import Tuple
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

    def _get_file_List(self, Pol):
        if Pol is not None:
            file_L = []
            name_L = []

            Pol_L = Pol.split('+')
            if 'HH' in Pol_L and self.HH_file is not None:
                file_L.append(self.HH_file)
                name_L.append(self.seen_id+'_HH')
            elif 'HH' in Pol_L and self.HH_file is None:
                print('HH file is not None')
            if 'HV' in Pol_L and self.HV_file is not None:
                file_L.append(self.HV_file)
                name_L.append(self.seen_id+'_HV')
            elif 'HV' in Pol_L and self.HV_file is None:
                print('HV file is not None')
            if 'VV' in Pol_L and self.VV_file is not None:
                file_L.append(self.VV_file)
                name_L.append(self.seen_id+'_VV')
            elif 'VV' in Pol_L and self.VV_file is None:
                print('VV file is not None')
            if 'VH' in Pol_L and self.VH_file is not None:
                file_L.append(self.VH_file)
                name_L.append(self.seen_id+'_VH')
            elif 'VH' in Pol_L and self.VH_file is None:
                print('VH file is not None')
        else:
            file_L = [self.HH_file, self.HV_file, self.VV_file, self.VH_file]
            name_L = [self.seen_id+'HH', self.seen_id +
                      'HV', self.seen_id+'VV', self.seen_id+'VH']

        return file_L, name_L

    def get_gcp_three(self):
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

    def get_conv_coef(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        return np.dot(self.coefficient_lon, lp_matrix), np.dot(self.coefficient_lat, lp_matrix)

    def make_intensity_fig(self, Pol_file, s_x=0, s_y=0, w=None, h=None, folder=None):
        if s_x < 0 or s_x > self.ncell:
            print('input error')
            exit()

        if s_y < 0 or s_y > self.nline:
            print('input error')
            exit()

        if w is None:
            w = self.ncell-s_x
        elif w > self.ncell-s_x:
            print('input error')
            exit()

        if h is None:
            h = self.nline-s_y
        elif h > self.nline-s_y:
            print('input error')
            exit()

        sigma, phase = self.get_intensity(Pol_file, s_x, s_y, w, h)
        name = str(self.seen_id)+'_' + \
            Pol_file.split('-')[2]+'_'+str(s_y)+'-'+str(s_x)
        if folder is None:
            plt.imsave(os.path.join(self.folder, name) +
                       '_sigma.png', sigma, cmap='gray')
            plt.imsave(os.path.join(self.folder, name) +
                       '_phase.png', phase, cmap='jet')
        else:
            plt.imsave(os.path.join(folder, name) +
                       '_sigma.png', sigma, cmap='gray')
            plt.imsave(os.path.join(folder, name) +
                       '_phase.png', phase, cmap='jet')

        return sigma, phase

    def get_intensity(self, Pol_file, s_x=0, s_y=0, w=None, h=None):
        if s_x < 0 or s_x > self.ncell:
            print('input error')
            exit()

        if s_y < 0 or s_y > self.nline:
            print('input error')
            exit()

        if w is None:
            w = self.ncell-s_x
        elif w > self.ncell-s_x:
            print('input error')
            exit()

        if h is None:
            h = self.nline-s_y
        elif h > self.nline-s_y:
            print('input error')
            exit()

        nrec = 544+self.ncell*8

        with open(Pol_file, mode='rb') as fp:
            fp.seek(720+int(nrec*s_y))
            data = struct.unpack(
                ">%s" % (int((nrec*h)/4))
                + "f", fp.read(int(nrec*h)))
            data = np.array(data).reshape(-1, int(nrec/4))
            data = data[:, int(544/4):int(nrec/4)]
            slc = data[:, ::2] + 1j*data[:, 1::2]
            slc = slc[:, s_x:s_x+w]

        sigma = 20*np.log10(abs(slc))+self.CF-32.0
        sigma = np.array(255*(sigma-np.amin(sigma)) /
                         (np.amax(sigma)-np.amin(sigma)), dtype="uint8")
        phase = np.angle(slc)
        phase = np.array(255*(phase - np.amin(phase)) /
                         (np.amax(phase) - np.amin(phase)), dtype="uint8")

        return sigma, phase
