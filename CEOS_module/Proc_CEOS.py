import itertools
import os
import struct
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


class Proc_CEOS:
    v21_colors = ['#ffffff', '#000064', '#ff0000', '#0080ff', '#ffc1bf',
                  '#ffff00', '#80ff00', '#00ff80', '#56ac00', '#00ac56',
                  '#806400', '#D9F003', '#A22978']
    v2103cmap = ListedColormap(v21_colors, name='ver2103')
    v18_colors = ['#ffffff', '#000064', '#ff0000', '#0080ff', '#ffc1bf',
                  '#ffff00', '#80ff00', '#00ff80', '#56ac00', '#00ac56', '#806400']
    v1803cmap = ListedColormap(v18_colors, name='ver2803')

    def __init__(self, folder) -> None:
        self.folder = folder
        self.HH_file = self.HV_file = self.VV_file = self.VH_file \
            = self.LED_file = self.nline = self.ncell = None
        self.GT_PATH = ''
        self.GT_FILE_LIST = {}
        self.cmap = None

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
                f.seek(1040)
                self.origin_lat = float(f.read(20))
                self.origin_lon = float(f.read(20))
        else:
            print('LED file connot be found.')
            exit()

    def __get_filename(self, folder, name) -> str:
        if folder is None:
            return os.path.join(self.folder, name)
        else:
            return os.path.join(folder, name)

    def get_gcp_three(self, line) -> list:
        with open(str(self.main_file), mode='rb') as f:
            f.seek(int(720+(line)*(self.ncell*8+544)+192))
            f_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            f_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000

        return [[f_lon, f_lat], [m_lon, m_lat], [e_lon, e_lat]]

    def get_lon_lat(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        return np.dot(self.coefficient_lon, lp_matrix), np.dot(self.coefficient_lat, lp_matrix)

    def save_gcp(self, x, y, w, h, folder=None) -> None:
        filename = self.seen_id+'-'+str(y)+'-'+str(x)+'.points'
        filename = self.__get_filename(folder, filename)

        with open(filename, mode='w') as f:
            s = ""
            x_l = np.linspace(x, x+w, 5, dtype='int')
            y_l = np.linspace(y, y+h, 5, dtype='int')

            for _x, _y in itertools.product(x_l, y_l):
                s += " -gcp "
                lon, lat = self.get_lon_lat(x+_x, y+_y)
                s += " ".join(
                    [str(_x), str(_y), str(lon), str(lat)])
            f.write(s)

    def save_intensity_OverAllfig(self, Pol_file, folder=None) -> None:
        img = np.empty((self.nline, self.ncell), dtype='float32')
        for h in range(self.nline):
            img[h], _ = self.get_intensity(
                Pol_file, 0, h, self.ncell, 1)

        img = np.array(255*(img-np.amin(img)) /
                       (np.amax(img)-np.amin(img)), dtype="uint8")

        filename = str(self.seen_id)+'.png'
        filename = self.__get_filename(folder, filename)
        plt.imsave(filename, img, cmap='gray')

    def save_intensity_fig(self, Pol_file, x=0, y=0, w=None, h=None, folder=None) -> Tuple[np.ndarray, np.ndarray]:
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
        filename = str(self.seen_id)+'-' + str(y)+'-' + \
            str(x)+'__'+Pol_file.split('-')[2]

        filename = self.__get_filename(folder, filename)

        plt.imsave(filename+'_sigma.png', sigma_img, cmap='gray')
        plt.imsave(filename+'_pahse.png', phase_img, cmap='jet')

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

    def get_GT(self, lat, lon) -> int:
        filename = os.path.join(self.GT_PATH, 'LC_N' +
                                str(int(lat))+'E'+str(int(lon))+'.tif')
        if filename not in self.GT_FILE_LIST.keys():
            img = Image.open(filename)
            self.GT_FILE_LIST[filename] = np.array(img)

        h_l, w_l = self.GT_FILE_LIST[filename].shape

        h, w = (h_l-1)-int((lat-int(lat))/(1 / h_l)
                           ), int((lon-int(lon))/(1 / w_l))

        return self.GT_FILE_LIST[filename][h][w]

    def save_GT_img(self, GT, x, y, w, h, folder=None) -> None:
        GT = np.array(GT)
        GT = GT.reshape(h, w)
        GT = np.flipud(GT)
        GT = np.rot90(GT, -1)
        GT[GT == 255] = 0

        filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'GT.png'
        filename = self.__get_filename(folder, filename)

        if 'ver2103' in self.GT_PATH:
            plt.imsave(filename, GT,
                       cmap=self.v2103cmap, vmin=0, vmax=13)
        elif 'ver1803' in self.GT_PATH:
            plt.imsave(filename, GT,
                       cmap=self.v1803cmap, vmin=0, vmax=11)
        else:
            if not self.cmap:
                print('you need set cmap')
            else:
                plt.imsave(filename, GT,
                           cmap=self.cmap, vmin=0, vmax=self.cmap.N)

    def set_cmap(self, colors) -> None:
        self.cmap = ListedColormap(colors, name='custom')
