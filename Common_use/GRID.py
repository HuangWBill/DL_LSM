# -*- coding: utf-8 -*-
'''
遥感影像的读取与写入,归一化
读取：read_img   多通道时，得到的影像格式为（通道,宽，高）
     proj, geotrans, data = GRID.read_img('路径')
写入：write_img    多通道时，输入的新数据必须为（通道,宽，高）
     GRID.write_img('路径', proj, geotrans, 新的数据)  # 写数据
归一化：norm     输入影像数据，最小值，最大值，输出归一化后的影像数据
     新数据名 = GRID.norm（影像数据，最小值，最大值）
'''
from osgeo import gdal

# 读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    return im_proj, im_geotrans, im_data

# 写文件，以写成tif为例
def write_img(filename, im_proj, im_geotrans, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

# 对输入的影像进行归一化到0-1
def norm(img,min,max):
    img_new = (img-min) / (max-min)
    return img_new

# 判断是否是tif格式的文件
def file_filter(f):
    if f[-4:] in ['.tif']:
        return True
    else:
        return False

