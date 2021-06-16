import itertools
import os
import struct
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


class Proc_CEOS:
    """
    ALOS2 CEOS L1.1のデータを処理する． 
    """
    v21_colors = ['#ffffff', '#000064', '#ff0000', '#0080ff', '#ffc1bf',
                  '#ffff00', '#80ff00', '#00ff80', '#56ac00', '#00ac56',
                  '#806400', '#D9F003', '#A22978']
    __v2103cmap = ListedColormap(v21_colors, name='ver2103')
    v18_colors = ['#ffffff', '#000064', '#ff0000', '#0080ff', '#ffc1bf',
                  '#ffff00', '#80ff00', '#00ff80', '#56ac00', '#00ac56', '#806400']
    __v1803cmap = ListedColormap(v18_colors, name='ver2803')

    def __init__(self, folder) -> None:
        """
        Parameters
        ----------
        folder : str
            処理対象のフォルダパス

        Attributes
        --------
        folder : str
            デフォルトのフォルダパス
        HH_file : str
            HH偏波SARイメージのファイルパス
        HV_file : str
            HV偏波SARイメージのファイルパス
        VV_file : str
            VV偏波SARイメージのファイルパス
        VH_file : str
            VH偏波SARイメージのファイルパス
        GT_PATH : str
            GTのフォルダパス（高解像度土地利用土地被覆図など）
        GT_IMG_LIST : dic
            利用したGT画像のキャッシュ
        cmap : list
            カスタムのカラーマップ
        nline : int
            SARイメージのライン数（高さ）
        ncell : int
            SARイメージのピクセル数（幅）
        LED_file : str
            SARリーダのファイルパス
        seen_id : str
            処理対象データのシーンID
        coordinate_flag : bool
            緯度経度変換係数での誤差の有無
        v21_colors : list
            高解像度土地利用土地被覆図ver21のカラーリスト
        v18_colors : list
            高解像度土地利用土地被覆図ver18のカラーリスト
        """
        self.folder = folder
        self.HH_file = self.HV_file = self.VV_file \
            = self.VH_file = self.LED_file = None
        self.GT_PATH = ''
        self.GT_IMG_LIST = {}
        self.cmap = None
        self.__main_file = ''
        self.__coefficient_lat = None
        self.__diff_origin_lon = self.__diff_origin_lat = None
        self.__lon_func_x = self.__lon_func_y \
            = self.__lat_func_x = self.__lat_func_y = None

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
                self.__main_file = self.HH_file
        elif self.HV_file is not None:
            with open(self.HV_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.__main_file = self.HV_file
        elif self.VV_file is not None:
            with open(self.VV_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.__main_file = self.VV_file
        elif self.VH_file is not None:
            with open(self.VH_file, mode='rb') as f:
                f.seek(236)
                self.nline = int(f.read(8))
                f.seek(248)
                self.ncell = int(f.read(8))
                self.__main_file = self.VH_file
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
                self.__coefficient_lat = [float(f.read(20)) for _ in range(25)]
                self.coefficient_lon = [float(f.read(20)) for _ in range(25)]

        else:
            print('LED file connot be found.')
            exit()

        self.coordinate_flag = self.__set_coordinate_adjust_value()

    def __make_filepath(self, folder, name) -> str:
        if folder is None:
            return os.path.join(self.folder, name)
        else:
            return os.path.join(folder, name)

    def __normalization(self, x) -> np.ndarray:
        return (x-np.percentile(x, q=1))/(np.percentile(x, q=99)-np.percentile(x, q=1))

    def __leefilter(self, img, size) -> np.ndarray:
        img_mean = uniform_filter(img, (size, size))
        img_sqr_mean = uniform_filter(img**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = variance(img)

        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (img - img_mean)

        return img_output

    def __set_coordinate_adjust_value(self) -> bool:
        coef_lon, coef_lat = self.__get__coordinate(0, 0)
        self.origin_lon, self.origin_lat = \
            self.get_coordinate_three_points(0)[0]

        if (coef_lon-self.origin_lon) < 1.0e-5 and (coef_lat-self.origin_lat) < 1.0e-5:
            self.__lat_func_x = self.__lat_func_y = self.__lon_func_x \
                = self.__lon_func_y = np.poly1d([0, 0, 0])
            return False
        else:
            x = [0, self.ncell/2, self.ncell-1]
            y = [0, int(self.nline/2), self.nline-1]
            three_lonlat = np.array(
                [self.get_coordinate_three_points(_y) for _y in y]).reshape(9, 2)
            coef_latlon = np.array([self.__get__coordinate(_x, _y)
                                    for _y, _x in itertools.product(y, x)])

            diff = three_lonlat-coef_latlon
            self.__diff_origin_lon, self.__diff_origin_lat = diff[0][0], diff[0][1]
            f_diff, m_diff, e_diff = diff.reshape(3, 3, 2)

            f_diff_lon = f_diff[:, :1]-f_diff[0][0]
            f_diff_lat = f_diff[:, 1:2]-f_diff[0][1]

            m_diff_lon = m_diff[:, :1]-m_diff[0][0]
            m_diff_lat = m_diff[:, 1:2]-m_diff[0][1]

            e_diff_lon = e_diff[:, :1]-e_diff[0][0]
            e_diff_lat = e_diff[:, 1:2]-e_diff[0][1]

            diff_ave = np.ravel((f_diff_lon+m_diff_lon+e_diff_lon)/3)
            coeff = np.polyfit(x, diff_ave, 2)
            self.__lon_func_x = np.poly1d(coeff)

            diff_ave = np.ravel((f_diff_lat+m_diff_lat+e_diff_lat)/3)
            coeff = np.polyfit(x, diff_ave, 2)
            self.__lat_func_x = np.poly1d(coeff)

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
            self.__lon_func_y = np.poly1d(coeff)

            diff_ave = np.ravel((f_diff_lat+m_diff_lat+e_diff_lat)/3)
            coeff = np.polyfit(y, diff_ave, 2)
            self.__lat_func_y = np.poly1d(coeff)
            return True

    def __get__coordinate_adjust_value(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        adjust_lon = self.__diff_origin_lon+self.__lon_func_x(pixel) +\
            self.__lon_func_y(line)
        adjust_lat = self.__diff_origin_lat+self.__lat_func_x(pixel) +\
            self.__lat_func_y(line)

        return np.dot(self.coefficient_lon, lp_matrix)+adjust_lon, np.dot(self.__coefficient_lat, lp_matrix)+adjust_lat

    def __get__coordinate(self, pixel, line) -> Tuple[float, float]:
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        return np.dot(self.coefficient_lon, lp_matrix), np.dot(self.__coefficient_lat, lp_matrix)

    def save_gcp(self, x, y, w, h, folder=None, filename=None) -> None:
        """
        gdal_translateで利用するgcpオプションをテキスト出力

        Parameters
        ----------
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ
        folder : str
            保存先フォルダのパス
        filename : str
            保存データのファイル名
        """

        if not filename:
            filename = self.seen_id+'-'+str(y)+'-'+str(x)+'.points'
        filepath = self.__make_filepath(folder, filename)

        with open(filepath, mode='w') as f:
            s = ""
            x_l = np.linspace(x, x+w, 5, dtype='int')
            y_l = np.linspace(y, y+h, 5, dtype='int')

            for _x, _y in itertools.product(x_l, y_l):
                s += " -gcp "
                lon, lat = self.get_coordinate(x+_x, y+_y)
                s += " ".join(
                    [str(_x), str(_y), str(lon), str(lat)])
            f.write(s)

    def save_intensity_OverAllimg(self, Pol_file, folder=None, filename=None) -> None:
        """
        全体の強度画像を保存

        Parameters
        ----------
        Pol_file : str
            SARイメージのパス
        folder : str
            保存先フォルダのパス
        filename : str
            保存データのファイル名
        """

        img = np.empty((self.nline, self.ncell), dtype='float32')
        for h in range(self.nline):
            img[h], _ = self.get_intensity(
                Pol_file, 0, h, self.ncell, 1)

        img = np.array(255*(img-np.amin(img)) /
                       (np.amax(img)-np.amin(img)), dtype="uint8")

        if not filename:
            filename = str(self.seen_id)+'.png'
        filepath = self.__make_filepath(folder, filename)

        plt.imsave(filepath, img, cmap='gray')

    def save_intensity_img(self, Pol_file, x=0, y=0, w=None, h=None, folder=None, filename=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定の位置，大きさの強度画像と位相画像を保存．
        後方散乱強度と位相を出力

        Parameters
        ----------
        Pol_file : str
            SARイメージのパス
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ
        folder : str
            保存先フォルダのパス
        filename : str
            保存データのファイル名

        Returns
        -------
        sigma : np.ndarray
            後方散乱強度
        phase : np.ndarray
            位相
        """

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
        sigma_img = self.__leefilter(sigma_img, 9)
        phase_img = np.array(255*(phase - np.amin(phase)) /
                             (np.amax(phase) - np.amin(phase)), dtype="uint8")
        phase_img = self.__leefilter(phase_img, 9)

        if not filename:
            filename = str(self.seen_id)+'-' + str(y)+'-' + \
                str(x)+'__'+Pol_file.split('-')[2]
        filepath = self.__make_filepath(folder, filename)

        plt.imsave(filepath+'_sigma.png', sigma_img, cmap='gray')
        plt.imsave(filepath+'_pahse.png', phase_img, cmap='jet')

        return sigma, phase

    def save_GT_img(self, GT, x, y, w, h, folder=None, filename=None) -> None:
        """
        GT画像の作成

        Parameters
        ----------
        GT : list
            GTデータの配列
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ
        folder : str
            保存先フォルダのパス
        filename : str
            保存データのファイル名
        """
        GT = np.array(GT)
        GT = GT.reshape(h, w)
        GT = np.flipud(GT)
        GT = np.rot90(GT, -1)
        GT[GT == 255] = 0

        if not filename:
            filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'GT.png'
        filepath = self.__make_filepath(folder, filename)

        if 'ver2103' in self.GT_PATH:
            plt.imsave(filepath, GT,
                       cmap=self.__v2103cmap, vmin=0, vmax=13)
        elif 'ver1803' in self.GT_PATH:
            plt.imsave(filepath, GT,
                       cmap=self.__v1803cmap, vmin=0, vmax=11)
        else:
            if not self.cmap:
                print('you need set cmap')
            else:
                plt.imsave(filepath, GT,
                           cmap=self.cmap, vmin=0, vmax=self.cmap.N)

    def save_Pauli_img(self, x, y, w, h, folder=None, filename=None) -> None:
        """
        Pauli分解画像の保存

        Parameters
        ----------
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ
        folder : str
            保存先フォルダのパス
        filename : str
            保存データのファイル名
        """
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

        if not filename:
            filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'Pauli.png'
        filepath = self.__make_filepath(folder, filename)

        plt.imsave(filepath, img)

    def get_intensity(self, Pol_file, x, y, w, h) -> Tuple[np.ndarray, np.ndarray]:
        """
        後方散乱強度と位相を出力

        Parameters
        ----------
        Pol_file : str
            SARイメージのパス
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ

        Returns
        -------
        sigma : np.ndarray
            後方散乱強度
        phase : np.ndarray
            位相
        """
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

    def get_slc(self, Pol_file, x, y, w, h) -> np.ndarray:
        """
        シグナルデータの取得

        Parameters
        ----------
        Pol_file : str
            SARイメージのパス
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ

        Returns
        -------
        slc : np.ndarray
            シグナルデータ（複素数）
        """
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

    def get_Pauli(self, x, y, w, h) -> np.ndarray:
        """
        Pauli分解画像の保存

        Parameters
        ----------
        x : int
            横方向の開始位置
        y : int
            縦方向の開始位置
        w : int
            幅
        h : int
            高さ
        """
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

        return r, g, b

    def get_coordinate_three_points(self, line) -> Tuple[list, list, list]:
        """
        指定のライン（高さ）の最初，中央，最後のピクセルの緯度経度の出力

        Parameters
        ----------
        line : int
            縦方向の位置

        Returns
        -------
        [f_lon, f_lat]: list
            最初のピクセルの緯度経度
        [m_lon, m_lat]: list
            中央のピクセルの緯度経度
        [e_lon, e_lat]: list
            最後のピクセルの緯度経度
        """
        with open(str(self.__main_file), mode='rb') as f:
            f.seek(int(720+(line)*(self.ncell*8+544)+192))
            f_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lat = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            f_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            m_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000
            e_lon = float(struct.unpack(">%s" % "i", f.read(4))[0])/1000000

        return [f_lon, f_lat], [m_lon, m_lat], [e_lon, e_lat]

    def get_coordinate(self, pixel, line) -> Tuple[float, float]:
        """
        指定の位置の緯度経度の出力

        Parameters
        ----------
        pixel : int
            横方向の位置
        line : int
            縦方向の位置

        Returns
        -------
        lon : float
            経度
        lat : float
            緯度
        """
        if self.coordinate_flag:
            return self.__get__coordinate_adjust_value(pixel, line)
        else:
            return self.__get__coordinate(pixel, line)

    def get_GT(self, lat, lon) -> int:
        """
        指定の位置のGTデータの取得

        Parameters
        ----------
        lat : float
            緯度
        lon : float
            経度

        Returns
        -------
        GT : int
            GTのクラス
        """
        filepath = os.path.join(self.GT_PATH, 'LC_N' +
                                str(int(lat))+'E'+str(int(lon))+'.tif')
        if filepath not in self.GT_IMG_LIST.keys():
            img = Image.open(filepath)
            self.GT_IMG_LIST[filepath] = np.array(img)

        h_l, w_l = self.GT_IMG_LIST[filepath].shape

        h, w = (h_l-1)-int((lat-int(lat))/(1 / h_l)
                           ), int((lon-int(lon))/(1 / w_l))

        return self.GT_IMG_LIST[filepath][h][w]

    def set_cmap(self, colors) -> None:
        """
        カスタムのカラーマップの設定

        Parameters
        ----------
        colors : list
        色の配列
        """
        self.cmap = ListedColormap(colors, name='custom')

    def set_GT(self, folder) -> None:
        """
        GTフォルダの設定

        Parameters
        ----------
        folder : str
        GTフォルダのパス
        """
        self.GT_PATH = folder
