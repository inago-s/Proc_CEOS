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
from scipy import interpolate
import pyproj


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
        self.DEM_PATH = ''
        self.GT_IMG_LIST = {}
        self.DEM_IMG_LIST = {}
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
                self.__coefficient_lon = [float(f.read(20)) for _ in range(25)]
                self.__P0 = float(f.read(20))
                self.__L0 = float(f.read(20))

        else:
            print('LED file connot be found.')
            exit()

    def __make_filepath(self, folder, name) -> str:
        if folder is None:
            return os.path.join(self.folder, name)
        else:
            return os.path.join(folder, name)

    def __normalization(self, x) -> np.ndarray:
        x = (x-np.percentile(x, q=1)) / \
            (np.percentile(x, q=99)-np.percentile(x, q=1))
        return np.clip(x, 0, 1)

    def __leefilter(self, img, size) -> np.ndarray:
        img_mean = uniform_filter(img, (size, size))
        img_sqr_mean = uniform_filter(img**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = variance(img)

        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (img - img_mean)

        return img_output

    def __get_sat_pos(self, y, h):
        led = open(self.LED_file, mode='rb')
        img = open(self.__main_file, mode='rb')
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
        time_obs = np.arange(time_start, time_end,
                             (time_end - time_start)/nline)
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
        pos = np.dstack((X, Y, Z))

        return pos[0][y:y+h, :]

    def adjust_lonlat(self, lonlat, y, h, geo):
        xyz2latlon = pyproj.Transformer.from_crs(7789, 4326)
        latlon2xyz = pyproj.Transformer.from_crs(4326, 7789)

        sat_pos = self.__get_sat_pos(y, h)
        w = int(len(lonlat)/h)
        new_lonlat = []
        for i in range(len(sat_pos)):
            sat_latlon = xyz2latlon.transform(
                sat_pos[i][0], sat_pos[i][1], sat_pos[i][2])
            for ll in lonlat[i*w:i*w+w]:
                h = self.get_DEM(ll[0], ll[1]) + geo
                obt_pos = latlon2xyz.transform(ll[1], ll[0], 0)
                sl = np.linalg.norm(obt_pos-sat_pos[i])
                gr_ob = np.sqrt(sl**2-sat_latlon[2]**2)
                gr_true = np.sqrt(sl**2-(sat_latlon[2]-h)**2)
                dis = np.linalg.norm(sat_pos[i][:2]-obt_pos[:2])
                diff = gr_true-gr_ob
                lat, lon = (ll[1]-sat_latlon[0])*(diff/dis) + ll[1],\
                    (ll[0]-sat_latlon[1])*(diff/dis)+ll[0]
                new_lonlat.append((lon, lat))

        return new_lonlat

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

    def save_DEM_img(self, DEM, x, y, w, h, folder=None, filename=None) -> None:
        """
        """
        DEM = np.array(DEM)
        DEM = DEM.reshape(h, w)
        DEM = np.flipud(DEM)
        DEM = np.rot90(DEM, -1)

        if not filename:
            filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'GT.png'
        filepath = self.__make_filepath(folder, filename)
        plt.imsave(filepath, DEM, cmap='jet')

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

    def save_coherency_img(self, x, y, w, h, folder=None, filename=None) -> None:
        HH = self.get_slc(self.HH_file, x, y, w, h)
        HV = self.get_slc(self.HV_file, x, y, w, h)
        VV = self.get_slc(self.VV_file, x, y, w, h)
        VH = self.get_slc(self.VH_file, x, y, w, h)
        HV = (VH+HV)/2

        double = np.zeros((h, w))
        surface = np.zeros((h, w))
        vol = np.zeros((h, w))
        helix = np.zeros((h, w))

        for x, y in itertools.product(range(w), range(h)):
            _HH = HH[y, x]
            _VV = VV[y, x]
            _HV = HV[y, x]
            Pc = 2*np.abs(((_HV.conjugate())*(_HH-_VV)).imag)
            if 10*np.log10((np.abs(_VV)**2)/(np.abs(_HH)**2)) < -2:
                Pv = (15/2)*(np.abs(_HV)**2)-(15/8)*Pc
                if Pv < 0:
                    Pc = 0
                    Pd = (15/2)*(np.abs(_HV)**2)-(15/8)*Pc
                S = (1/2)*(np.abs(_HH+_VV)**2)-(Pv/2)
                D = (1/2)*(np.abs(_HH-_VV)**2)-(7/4)*(np.abs(_HV)**2)-(Pc/16)
                C = (1/2)*((_HH+_VV)*((_HH-_VV).conjugate()))-(Pv/6)
            elif 10*np.log10((np.abs(_VV)**2)/(np.abs(_HH)**2)) > 2:
                Pv = (15/2)*(np.abs(_HV)**2)-(15/8)*Pc
                if Pv < 0:
                    Pc = 0
                    Pv = (15/2)*(np.abs(_HV)**2)-(15/8)*Pc
                S = (1/2)*(np.abs(_HH+_VV)**2)-(Pv/2)
                D = (1/2)*(np.abs(_HH-_VV)**2)-(7/4)*(np.abs(_HV)**2)-(Pc/16)
                C = (1/2)*((_HH+_VV)*((_HH-_VV).conjugate()))+(Pv/6)
            else:
                Pv = 8*(np.abs(_HV)**2)-2*Pc
                if Pv < 0:
                    Pc = 0
                    Pv = 8*(np.abs(_HV)**2)-2*Pc
                S = (1/2)*(np.abs(_HH+_VV)**2)-4*(np.abs(_HV)**2)+Pc
                D = (1/2)*(np.abs(_HH-_VV)**2)-2*(np.abs(_HV)**2)
                C = (1/2)*((_HH+_VV)*((_HH-_VV).conjugate()))
            TP = np.abs(_HH)**2+np.abs(_VV)**2+np.abs(_HV)

            if Pv < TP or Pv < TP:
                C0 = _HH*(_VV.conjugate())-np.abs(_HV)**2+Pc/2
                if C0.real > 0:
                    Ps = S+np.abs(C)**2/S
                    Pd = D-np.abs(C)**2/S
                else:
                    Ps = S-np.abs(C)**2/D
                    Pd = D+np.abs(C)**2/D
                if Ps > 0 and Pd < 0:
                    Ps = TP-Pv-Pc
                    Pd = 0
                elif Ps < 0 and Pd > 0:
                    Pd = TP-Pv-Pc
                    Ps = 0
            else:
                Ps = Pd = 0
                Pv = TP-Pc

            double[y, x] = Pd
            surface[y, x] = Ps
            vol[y, x] = Pv
            helix[y, x] = Pc

        surface[surface == 0] = 1
        double[double == 0] = 1
        helix[helix == 0] = 1
        sigma = 10*np.log10(abs(surface))-self.CF-32
        sigma = (self.__normalization(sigma)*255).astype('uint8')
        fs = self.__leefilter(cv2.equalizeHist(sigma), 9).astype('uint8')

        sigma = (10*np.log10(abs(double))-self.CF-32) + \
            (10*np.log10(abs(helix))-self.CF-32)/2
        sigma = (self.__normalization(sigma)*255).astype('uint8')
        fd = self.__leefilter(cv2.equalizeHist(sigma), 9).astype('uint8')

        sigma = (10*np.log10(abs(vol))-self.CF-32) + \
            (10*np.log10(abs(helix))-self.CF-32)/2
        sigma = (self.__normalization(sigma)*255).astype('uint8')
        fv = self.__leefilter(cv2.equalizeHist(sigma), 9).astype('uint8')

        if not filename:
            filename = self.seen_id+'-'+str(y)+'-'+str(x)+'__' + 'Pauli.png'
        filepath = self.__make_filepath(folder, filename)

        img = np.dstack((fd, fv, fs))

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
        pixel = pixel-self.__P0
        line = line-self.__L0
        l_matrix = np.array([line**4, line**3, line**2, line**1, 1])
        p_matrix = np.array(
            [[pixel**4], [pixel**3], [pixel**2], [pixel**1], [1]])
        lp_matrix = np.ravel(l_matrix*p_matrix)

        return np.dot(self.__coefficient_lon, lp_matrix), np.dot(self.__coefficient_lat, lp_matrix)

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

    def get_GT(self, lon, lat) -> int:
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

    def get_DEM(self, lon, lat) -> int:
        filepath = os.path.join(self.DEM_PATH, 'ALPSMLC30_N' +
                                str(int(lat)).zfill(3)+'E'+str(int(lon)).zfill(3)+'_DSM.tif')

        if filepath not in self.DEM_IMG_LIST.keys():
            img = Image.open(filepath)
            self.DEM_IMG_LIST[filepath] = np.array(img)

        h_l, w_l = self.DEM_IMG_LIST[filepath].shape

        h, w = (h_l-1)-int((lat-int(lat))/(1 / h_l)
                           ), int((lon-int(lon))/(1 / w_l))

        return self.DEM_IMG_LIST[filepath][h][w]

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

    def set_DEM(self, folder) -> None:
        """
        DEMフォルダの設定

        Parameters
        ----------
        folder : str
        DEMフォルダのパス
        """
        self.DEM_PATH = folder
