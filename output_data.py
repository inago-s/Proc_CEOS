from CEOS_module.Proc_CEOS import Proc_CEOS
import itertools
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm


def main():
    C = Proc_CEOS('../ALOS2024040390-141102')
    colors = ['#ffffff', '#000064', '#ff0000', '#0080ff', '#ffc1bf',
              '#ffff00', '#ff6533', '#806400', '#05fc81', '#007c3d']

    C.set_cmap(colors)
    C.set_GT('../ver1609VT15_LC_GeoTiff/')

    filename = C.seen_id+'.png'

    latlon = Parallel(n_jobs=-1)(delayed(C.get_coordinate)
                                 (s_x, s_y)for s_x, s_y in itertools.product(range(C.ncell), range(C.nline)))

    class_num = Parallel(n_jobs=-1, require='sharedmem')(delayed(C.get_GT)
                                                         (ll[1], ll[0])for ll in latlon)

    C.save_GT_img(class_num, 0, 0, C.ncell, C.nline, 'demo-GT', filename)
    Pol_L = [C.HH_file, C.HV_file, C.VV_file, C.VH_file]
    n = len(Pol_L)

    h, w = 1024, 1024

    s_x_list = np.array([i*w for i in range(int(C.ncell/w))])
    s_y_list = np.array([i*h for i in range(int(C.nline/h))])

    # s_x_list = [2000]
    # s_y_list = [15000]

    for x, y in tqdm(itertools.product(s_x_list, s_y_list)):
        filename = C.seen_id+'-'+str(y)+'-'+str(x)+'.png'
        C.save_Pauli_img(x, y, w, h, 'demo-image', filename)

    latlon = Parallel(n_jobs=-1)(delayed(C.get_coordinate)
                                 (s_x, s_y)for s_x, s_y in itertools.product(range(x, x+h), range(y, y+h)))

    class_num = Parallel(n_jobs=-1, require='sharedmem')(delayed(C.get_GT)
                                                         (ll[1], ll[0])for ll in latlon)

    C.save_GT_img(class_num, x, y, w, h, 'demo-GT', filename)


if __name__ == '__main__':
    main()
