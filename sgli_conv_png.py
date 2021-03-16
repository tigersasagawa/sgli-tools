import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import cv2

parser = argparse.ArgumentParser(description='This .py file converts hdf file of SGLI to png file.')
parser.add_argument('input', help='file path of input')
#parser.add_argument('output', help='file path of output')
args = parser.parse_args()

input_name = args.input
file_name = input_name.split('/')[-1]
#output = args.output

v = int(file_name[21:23])
h = int(file_name[23:25])

def latlon2index_eqr(latlon):
    lat = latlon[0]
    lon = latlon[1]
    row = (ulat-lat)/d-0.5
    col = (lon-llon)/d-0.5
    return np.vstack([row, col])

def index_sin2latlon(pos, ij):
    i = ij[0]
    j = ij[1]
    lat = 90-10*pos[0]-(i+0.5)/480
    lon = (-180+10*pos[1]+j/480)/np.cos(np.deg2rad(lat))
    return np.vstack([lat, lon])

d = 1/480

file = h5py.File(input_name, 'r')

ullat, ullon = index_sin2latlon([v, h], [-0.5, -0.5])
lllat, lllon = index_sin2latlon([v, h], [4799.5, -0.5])
llon = np.min([lllon, ullon])
llat = lllat
lrlat, lrlon = index_sin2latlon([v, h], [4799.5, 4799.5])
urlat, urlon = index_sin2latlon([v, h], [-0.5, 4799.5])
rlon = np.max([lrlon, urlon])
ulat = urlat

llon = np.round(llon/d)*d #We should remove here if we don't need to conect other tile.
rlon = np.round(rlon/d)*d #We should remove here if we don't need to conect other tile.

row_num = 4800
col_num = np.int(np.abs(np.round((llon-rlon)/d)))

QA_flag = file['Image_data']['QA_flag']
Rs_VN03 = file['Image_data']['Rs_VN03']
Rs_VN05 = file['Image_data']['Rs_VN05']
Rs_VN08 = file['Image_data']['Rs_VN08']

qa = QA_flag[()]*QA_flag.attrs['Slope']+QA_flag.attrs['Offset']
vn03, vn05, vn08 = Rs_VN03[()], Rs_VN05[()], Rs_VN08[()]
vn03 = np.where(qa == 2, vn03, np.nan)
vn05 = np.where(qa == 2, vn05, np.nan)
vn08 = np.where(qa == 2, vn08, np.nan)
data = np.dstack([vn03, vn05, vn08]).reshape(4800**2, 3)

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
data[:, 2] = data[:, 2]*Rs_VN08.attrs['Slope']+Rs_VN08.attrs['Offset']
data = data.reshape(4800, 4800, 3)
data = np.hstack([np.nan*np.zeros([4800, 1, 3]), data, np.nan*np.zeros([4800, 1, 3])])
data = 2250*data #not needed. if we want geotiff.

num_list = np.arange(0, 4800*4801)
row_list = np.hstack([np.arange(0, 4800).reshape(-1, 1), np.array(num_list//4801).reshape(4800, 4801)]).flatten()
col_list = np.hstack([-1*np.ones(4800).reshape(-1, 1), np.fmod(num_list, 4801).reshape(4800, 4801)]).flatten()

rowcol_sin = np.vstack([row_list, col_list])
latlon = index_sin2latlon([v, h], rowcol_sin)
rowcol_eqr = latlon2index_eqr(latlon)
col_eqr = rowcol_eqr[1].reshape(4800, 4802)

output_array = []
for i in tqdm(range(0, 4800)):
    extractor = NearestNeighbors(metric='euclidean', n_neighbors=1)
    extractor.fit(col_eqr[i].reshape(-1, 1))
    col = extractor.kneighbors(np.arange(0, col_num).reshape(-1, 1))[1].flatten()
    output_array.append(data[i][col])
output_array = np.array(output_array)

#year = file_name[7:11]
#dir_name = './images_'+year+'_'+file_name[21:23]+file_name[23:25]+'/'
#output_name = dir_name+file_name[:-3]+'.png'
file_name = file_name[:-3]+'.png'
cv2.imwrite(file_name, output_array)
