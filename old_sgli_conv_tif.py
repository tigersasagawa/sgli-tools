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
file_name = input_name.split('/')[-1]
#output = args.output

def progress(x):
    print("\r["+"#"*x+" "*(10-x)+"]", end="")
    print(format(x, '02'), end="")
    print("/10", end="")
    
progress(1)

def index_eqr2latlon(rowcol):
    row = rowcol[0]
    col = rowcol[1]
    lat = ulat-(row+0.5)*d
    lon = llon+(col+0.5)*d
    return np.vstack([lat, lon])

def latlon2index_sin(tile_pos, latlon):
    lat = latlon[0]
    lon = latlon[1]
    pixel_num_total = 360/d
    pixel_num_part = pixel_num_total*np.cos(np.deg2rad(lat))
    row = (90-lat)/d-tile_pos[0]*4800
    col = pixel_num_part*(lon/360)+pixel_num_total/2-tile_pos[1]*4800
    return np.vstack([row, col])

d = 1/480
nan = [np.nan for _ in range(0, 11)]

file = h5py.File(input_name, 'r')

progress(2)

lllon = file['Geometry_data'].attrs['Lower_left_longitude'][0]
ullon = file['Geometry_data'].attrs['Upper_left_longitude'][0]
llon = np.min([lllon, ullon])

lrlon = file['Geometry_data'].attrs['Lower_right_longitude'][0]
urlon = file['Geometry_data'].attrs['Upper_right_longitude'][0]
rlon = np.max([lrlon, urlon])

llat = file['Geometry_data'].attrs['Lower_left_latitude'][0]
ulat = file['Geometry_data'].attrs['Upper_left_latitude'][0]

#llon = np.int(llon/d)*d #We should remove here if we don't need to conect other tile.
#rlon = np.int(rlon/d)*d #We should remove here if we don't need to conect other tile.

row_num = 4800
col_num = np.int(np.abs(np.round((llon-rlon)/d)))

QA_flag = file['Image_data']['QA_flag']
qa = QA_flag[()]*QA_flag.attrs['Slope']+QA_flag.attrs['Offset']

data = []
for band in np.arange(0, 11):
    Rs_VNXX = file['Image_data']['Rs_VN'+format(band+1, '02')]
    vn_XX = Rs_VNXX[()]
    #vn_XX = np.where(qa == 2, vn_XX, np.nan)
    data.append(vn_XX)
    
data = np.array(data).transpose(1, 2, 0)
data = data.reshape(4800**2, 11)

progress(3)

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

progress(4)

for band in np.arange(0, 11):
    slope = file['Image_data']['Rs_VN'+format(band+1, '02')].attrs['Slope']
    offset = file['Image_data']['Rs_VN'+format(band+1, '02')].attrs['Offset']
    data[:, band] = data[:, band]*slope+offset
    
data = data.reshape(4800, 4800, 11)
#data = 2250*data #not needed. if we want geotiff.

v = int(file_name[21:23])
h = int(file_name[23:25])

progress(5)

num_list = np.arange(0, row_num*col_num)
row_list = num_list//col_num
col_list = num_list%col_num

progress(6)

rowcol_eqr = np.dstack([row_list, col_list]).reshape(row_num*col_num, 2).T
latlon = index_eqr2latlon(rowcol_eqr)
rowcol_sin = latlon2index_sin([v, h], latlon)

progress(7)

row_sin = rowcol_sin[0]
col_sin = rowcol_sin[1]
row_sin = row_sin.astype(np.int64)
col_sin = col_sin.astype(np.int64)

mask = np.where((0 <= col_sin) & (col_sin < 4800), 0, np.nan)
col_sin = np.where((0 <= col_sin) & (col_sin < 4800), col_sin, 0)

progress(8)

output_array = data[row_sin, col_sin]
output_array = (output_array.T+mask).T
output_array = output_array.reshape(row_num, col_num, 11)

progress(9)

#year = file_name[7:11]
#dir_name = './images_'+year+'/'
output_name = file_name[:-3]+'.tif'
output = gdal.GetDriverByName('GTiff').Create(output_name, col_num, row_num, 11, gdal.GDT_Float64)
output.SetGeoTransform((llon, d, 0, ulat, 0, -d))
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
output.SetProjection(srs.ExportToWkt())
for i in range(0, 11):
    output.GetRasterBand(i+1).WriteArray(output_array[:, :, i])
output.FlushCache()
output = None

progress(10)
