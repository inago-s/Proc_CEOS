import itertools
import os
import struct
from typing import Tuple
import cv2
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

        else:
            print('LED file connot be found.')
            exit()

        self.diff_origin_lon = self.diff_origin_lat = None
        self.lon_func_x = self.lon_func_y \
            = self.lat_func_x = self.lat_func_y = None

        self.coordinate_flag = self.__set_coordinate_adjust_value()

    def __make_filepath(self, folder, name) -> str:
        if folder is None:
            return os.path.join(self.folder, name)
        else:
            return os.path.join(folder, name)

    def __normalization(self, x) -> np.ndarray:
        return (x-np.percentile(x, q=1))/(np.percentile(x, q=99)-np.percentile(x, q=1))

    def __set_coordinate_adjust_value(self) -> bool:
        coef_lon, coef_lat = self.__get__coordinate(0, 0)
        self.origin_lon, self.origin_lat = \
            self.get_coordinate_three_points(0)[0]

        if (coef_lon-self.origin_lon) < 1.0e-5 and (coef_lat-self.origin_lat) < 1.0e-5:
            self.lat_func_x = self.lat_func_y = self.lon_func_x \
                = self.lon_func_y = np.poly1d([0, 0, 0])
            return False
        else:
            x = [0, self.ncell/2, self.ncell-1]
            y = [0, int(self.nline/2), self.nline-1]
            three_lonlat = np.array(
                [self.get_coordinate_three_points(_y) for _y in y]).reshape(9, 2)
            coef_latlon = np.array([self.__get__coordinate(_x, _y)
                                    for _y, _x in itertools.product(y, x)])

            diff = three_lonlat-coef_latlon
            self.diff_origin_lon, self.diff_origin_lat = diff[0][0], diff[0][1]
            f_diff, m_diff, e_diff = diff.reshape(3, 3, 2)

            f_diff_lon = f_diff[:, :1]-f_diff[0][0]
            f_diff_lat = f_diff[:, 1:2]-f_diff[0][1]

            m_diff_lon = m_diff[:, :1]-m_diff[0][0]
            m_diff_lat = m_diff[:, 1:2]-m_diff[0][1]

            e_diff_lon = e_diff[:, :1]-e_diff[0][0]
            e_diff_lat = e_diff[:, 1:2]-e_diff[0][1]

            diff_ave = np.ravel((f_diff_lon+m_diff_lon+e_diff_lon)/3)
            coeff = np.polyfit(x, diff_ave, 2)
            self.lon_func_x = np.poly1d(coeff)

            diff_ave = np.ravel((f_diff_lat+m_diff_lat+e_diff_lat)/3)
            coeff = np.polyfit(x, diff_ave, 2)
            self.lat_func_x = np.poly1d(coeff)

            f_diff, m_diff, e_diff = diff[::3, ], diff[1::3, ], diff[2::3, ]

            f_diff_lon = f_diff[:, :1]-f_diff[0][0]
            f_diff_lat = f_diff[:, 1:2]-f_diff[0][1]

            m_diff_lon = m_diff[:, :1]-m_diff[0][0]
            m_diff_lat = m_diff[:, 1:2]-m_diff[0][1]

            e_diff_lon = e_diff[:, :1]-e_diff[0][0]
            e_diff_lat = e_diff[:, 1:2]-e_diff[0][1]

            diff_ave = np.ravel((f_diff_lat+m_diff_lat+e_diff_lat)/3)
            diff_ave = np.ravel(diff_ave/3)
            coeff = np.polyfit(y, diff_ave, 2)
            self.lon_func_y = np.poly1d(coeff)

            diff_ave = np.ravel((f_diff_lat+m_diff_lat+e_diff_lat)/3)
            coeff = np.polyfit(y, diff_ave, 2)
            self.lat_func_y = np.poly1d(coeff)
            return True

    def __get__coordinate_adjust_value(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        adjust_lon = self.diff_origin_lon+self.lon_func_x(pixel) +\
            self.lon_func_y(line)
        adjust_lat = self.diff_origin_lat+self.lat_func_x(pixel) +\
            self.lat_func_y(line)

        return np.dot(self.coefficient_lon, lp_matrix)+adjust_lon, np.dot(self.coefficient_lat, lp_matrix)+adjust_lat

    def __get__coordinate(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        return np.dot(self.coefficient_lon, lp_matrix), np.dot(self.coefficient_lat, lp_matrix)

    def save_gcp(self, x, y, w, h, folder=None) -> None:
        filename = self.seen_id+'-'+str(y)+'-'+str(x)+'.points'
        filepath = self.__make_filepath(folder, filename)

        with open(filename, mode='w') as f:
            s = ""
            x_l = np.linspace(x, x+w, 5, dtype='int')
            y_l = np.linspace(y, y+h, 5, dtype='int')

            for _x, _y in itertools.product(x_l, y_l):
                s += " -gcp "
                lon, lat = self.get_coordinate(x+_x, y+_y)
                s += " ".join(
                    [str(_x), str(_y), str(lon), str(lat)])
            f.write(s)

    def save_intensity_OverAllimg(self, Pol_file, folder=None) -> None:
        img = np.empty((self.nline, self.ncell), dtype='float32')
        for h in range(self.nline):
            img[h], _ = self.get_intensity(
                Pol_file, 0, h, self.ncell, 1)

        img = np.array(255*(img-np.amin(img)) /
                       (np.amax(img)-np.amin(img)), dtype="uint8")

        filename = str(self.seen_id)+'.png'
        filepath = self.__make_filepath(folder, filename)
        plt.imsave(filepath, img, cmap='gray')

    def save_intensity_img(self, Pol_file, x=0, y=0, w=None, h=None, folder=None) -> Tuple[np.ndarray, np.ndarray]:
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

        filepath = self.__make_filepath(folder, filename)

        plt.imsave(filepath+'_sigma.png', sigma_img, cmap='gray')
        plt.imsave(filepath+'_pahse.png', phase_img, cmap='jet')

        return sigma, phase

    def save_GT_img(self, GT, x, y, w, h, folder=None) -> None:
        GT = np.array(GT)
        GT = GT.reshape(h, w)
        GT = np.flipud(GT)
        GT = np.rot90(GT, -1)
        GT[GT == 255] = 0

        filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'GT.png'
        filepath = self.__make_filepath(folder, filename)

        if 'ver2103' in self.GT_PATH:
            plt.imsave(filepath, GT,
                       cmap=self.v2103cmap, vmin=0, vmax=13)
        elif 'ver1803' in self.GT_PATH:
            plt.imsave(filepath, GT,
                       cmap=self.v1803cmap, vmin=0, vmax=11)
        else:
            if not self.cmap:
                print('you need set cmap')
            else:
                plt.imsave(filepath, GT,
                           cmap=self.cmap, vmin=0, vmax=self.cmap.N)

    def save_Pauli_img(self, x, y, h, w, folder=None) -> None:
        HH = self.get_slc(self.HH_file, x, y, w, h)
        HV = self.get_slc(self.HV_file, x, y, w, h)
        VV = self.get_slc(self.VV_file, x, y, w, h)
        VH = self.get_slc(self.VH_file, x, y, w, h)

        r = 20*np.log10(abs((HH-VV)/np.sqrt(2)))+self.CF-32.0
        g = 20*np.log10(abs(np.sqrt(2)*((HV+VH)/2)))+self.CF-32.0
        b = 20*np.log10(abs((HH+VV)/np.sqrt(2)))+self.CF-32.0

        r = (self.__normalization(r)*255).astype('uint8')
        g = (self.__normalization(g)*255).astype('uint8')
        b = (self.__normalization(b)*255).astype('uint8')

        img = np.dstack([cv2.equalizeHist(r),
                         cv2.equalizeHist(g), cv2.equalizeHist(b)])

        filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'Pauli.png'
        filepath = self.__make_filepath(folder, filename)

        plt.imsave(filepath, img)

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

    def get_slc(self, Pol_file, x=0, y=0, w=None, h=None) -> np.ndarray:
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

        return slc

    def get_coordinate_three_points(self, line) -> Tuple[list, list, list]:
        with open(str(self.main_file), mode='rb') as f:
            f.seek(int(720+(line)*(self.ncell*8+544)+192))
            f_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            f_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000

        return [f_lon, f_lat], [m_lon, m_lat], [e_lon, e_lat]

    def get_coordinate(self, pixel, line) -> Tuple[float, float]:
        if self.coordinate_flag:
            return self.__get__coordinate_adjust_value(pixel, line)
        else:
            return self.__get__coordinate(pixel, line)

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

    def set_cmap(self, colors) -> None:
        self.cmap = ListedColormap(colors, name='custom')
