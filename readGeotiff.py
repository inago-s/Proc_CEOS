from osgeo import gdal
import osr
import sys


def main():
    if len(sys.argv) < 2:
        print("you need to enter the sar folder")
        exit()
    else:
        pic_name = sys.argv[1]

    ds = gdal.Open(pic_name)

    geotransform = ds.GetGeoTransform()
    print(5*geotransform[1]+geotransform[0])


if __name__ == "__main__":
    main()
