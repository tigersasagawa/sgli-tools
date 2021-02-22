import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from mpl_toolkits.basemap import Basemap


parser = argparse.ArgumentParser(description='This .py file cuts the latlon map (.png file) and displays the position of each observation sites.')
parser.add_argument('input', help='file path of input')
args = parser.parse_args()
input_name = args.input
file_name = input_name.split('/')[-1]

llon = 130.54166666666666
rlon = 171.12916666666666
ulat = 50
llat = 40

d = 1/480

name = [
  'TSE',
  'TOC']

lat = [
  45.055830,
  42.709720]

lon = [
  142.107145,
  141.565796]

for n in range(0, 2):
    print(name[n])
    image_data = np.flipud(cv2.imread(input_name)[:, :, [2, 1, 0]])
    fig = plt.figure(1, figsize=(5, 5))
    axes = fig.add_subplot(111)
    Map1 = Basemap(projection='cyl', resolution='i', llcrnrlon=lon[n]-50*d, llcrnrlat=lat[n]-50*d, urcrnrlon=lon[n]+50*d, urcrnrlat=lat[n]+50*d)
    Map2 = Basemap(projection='cyl', resolution='i', llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=rlon, urcrnrlat=ulat)
    Map2.imshow(image_data)
    axes.plot(lon[n], lat[n], markersize=12.5, color='red', marker='o')
    axes.set_xlim(lon[n]-50*d, lon[n]+50*d)
    axes.set_ylim(lat[n]-60*d, lat[n]+50*d)
    Map1.drawmeridians(np.arange(llon, rlon+0.1, 0.1), labels=[1, 1, 1, 1], dashes=[2, 2], linewidth=2, color='white')
    Map1.drawparallels(np.arange(llat, ulat+0.1, 0.1), labels=[1, 1, 1, 1], dashes=[2, 2], linewidth=2, color='white')
    output_name = './cut_img_2020_0428/'+file_name[:-4]+'_'+name[n]+'.png'
    plt.savefig(output_name, dpi=600)
    plt.close()

print('-----')
