import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from osgeo import gdal, osr

parser = argparse.ArgumentParser(description='This .py file converts hdf file of SGLI to png file.')
parser.add_argument('input', help='file path of input')
#parser.add_argument('output', help='file path of output')
args = parser.parse_args()

input_name = args.input
#output = args.output

def index_sin2latlon(tile_num, i, j):
    d = 1/480 # [deg/pixel] lat方向
    row_total = i+(tile_num[0]*4800) #total pixel above the pixel
    col_total = j+(tile_num[1]*4800) #total pixel left to the pixel
    lat = 90-(row_total+0.5)*d
    lon = 360*((col_total-180/d+0.5)/(np.cos(np.deg2rad(lat))*360/d))
    return [lat, lon]

def latlon2index_eqr(lat, lon):
    d_lat = np.abs(lat-llat)
    d_lon = np.abs(lon-llon)
    row = np.int(d_lat/d)
    col = np.int(d_lon/d)
    return [row, col]

def index_eqr2latlon(row, col):
    lat = (row+0.5)*d+llat
    lon = (col+0.5)*d+llon
    return [lat, lon]

def latlon2index_sin(tile_pos, lat, lon):
    d = 1/480
    pixel_num_total = 360/d
    pixel_num_part = pixel_num_total*np.cos(np.deg2rad(lat))
    row = (90-lat)/d-tile_pos[0]*4800
    col = pixel_num_part*(lon/360)+pixel_num_total/2-tile_pos[1]*4800
    return row, col

def NN(x, data_list):
    data_list = np.array(data_list)
    d = data_list-x
    index = np.argmin(np.abs(d))
    return index

d = 1/480
nan = [np.nan, np.nan, np.nan]

file = h5py.File(input_name, 'r')

lllon = file['Geometry_data'].attrs['Lower_left_longitude'][0]
ullon = file['Geometry_data'].attrs['Upper_left_longitude'][0]
llon = np.min([lllon, ullon])

lrlon = file['Geometry_data'].attrs['Lower_right_longitude'][0]
urlon = file['Geometry_data'].attrs['Upper_right_longitude'][0]
rlon = np.max([lrlon, urlon])

llat = file['Geometry_data'].attrs['Lower_left_latitude'][0]
ulat = file['Geometry_data'].attrs['Upper_left_latitude'][0]

llon = np.round(llon/d)*d #ここで端をまとめるのかそのままにするのかは, 選択できるようにしたほうがいいと思う.
rlon = np.round(rlon/d)*d #もし繋げたりしたいのなら, これは必要だし, 1タイルに注目するなら, ここは不要. 2021/02/06

row_num = 4800
col_num = np.int(np.abs(np.round(llon-rlon)/d))

QA_flag = file['Image_data']['QA_flag']
Rs_VN03 = file['Image_data']['Rs_VN03']
Rs_VN05 = file['Image_data']['Rs_VN05']
Rs_VN07 = file['Image_data']['Rs_VN07']

qa = QA_flag[()]*QA_flag.attrs['Slope']+QA_flag.attrs['Offset']
vn03, vn05, vn07 = Rs_VN03[()], Rs_VN05[()], Rs_VN07[()]
vn03 = np.where(qa == 2, vn03, np.nan)
vn05 = np.where(qa == 2, vn05, np.nan)
vn07 = np.where(qa == 2, vn07, np.nan)
data = np.dstack([vn03, vn05, vn07]).reshape(4800**2, 3)
condition = deepcopy(data)
judge = condition == 65535
condition[judge] = 1
condition[~judge] = 0
condition = np.sum(condition, axis=1).astype('float64')
judge = condition == 0
condition[judge] = 0
condition[~judge] = np.nan
condition = condition.reshape(4800**2, 1)
data = data+condition
data[:, 0] = data[:, 0]*Rs_VN03.attrs['Slope']+Rs_VN03.attrs['Offset']
data[:, 1] = data[:, 1]*Rs_VN05.attrs['Slope']+Rs_VN05.attrs['Offset']
data[:, 2] = data[:, 2]*Rs_VN07.attrs['Slope']+Rs_VN07.attrs['Offset']
data = data.reshape(4800, 4800, 3)

#data = 2000*data #cv2などで.png形式で表示する際にはおそらく必要. geotiffなら不要.

v = int(input_name[21:23])
h = int(input_name[23:25])

output_array = []
for pix in tqdm(range(0, row_num*col_num)):
    row_eqr = pix//col_num
    col_eqr = pix%col_num
    lat, lon = indexeqr2latlon(row_eqr, col_eqr)
    row_sin, col_sin = latlon2index_sin([v, h], lat, lon)
    if 0 <= col_sin <= 4799
        output_array.append(data[np.int(row_sin), np.int(col_sin)])
    else:
        output_array.append(nan)
output_array = np.array(output_array).reshape(row_num, col_num, 3)
output_name = input_name[:-3]+'.tif'
output = gdal.GetDriverByName('GTiff').Create(output_name, col_num, row_num, 3, gdal.GDT_Float64)
output.SetGeoTransform((llon, d, 0, ulat, 0, -d))
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
output.SetProjection(srs.ExportToWkt())
for i in range(0, 3):
    output.GetRasterBand(i+1).WriteArray(output_array[:, :, i])
output.FlushCache()
output = None
