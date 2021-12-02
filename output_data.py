from CEOS_module.Proc_CEOS import Proc_CEOS
import itertools
import numpy as np
from tqdm import tqdm


def main():
    # ALOS2 CEOSデータを読み込んだインスタンスを作成
    C = Proc_CEOS('')

    # GTフォルダパスの設定
    C.set_GT('')

    # DEMフォルダパスの設定
    C.set_DEM('')

    # 画像の縦横の大きさ指定
    h, w = 256, 256

    # ループ処理のため開始位置のリストを取得
    s_x_list = np.array([i*w for i in range(int(C.ncell/w))])
    s_y_list = np.array([i*h for i in range(int(C.nline/h))])

    bar = tqdm(total=len(s_x_list)*len(s_y_list))
    for x, y in tqdm(itertools.product(s_x_list, s_y_list)):
        # 強度画像（他の偏波は"HH"の部分を変更すれば良い）
        C.save_intensity_img('HH', x, y, w, h)

        # Pauli画像
        C.save_Pauli_img(x, y, w, h)

        # GTモノクロ画像
        C.save_GT_img(x, y, w, h, 'v21')
        bar.update(1)


if __name__ == '__main__':
    main()
