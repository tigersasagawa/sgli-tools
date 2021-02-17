import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from mpl_toolkits.basemap import Basemap


parser = argparse.Argumentparser('input', )

llon = 127.01666666666667
rlon = 156.64791666666667
ulat = 40
llat = 30

name = [
  'TSE',
  'TOC',
  'TKY',
  'MSE',
  'FJY',
  'FHK']

lat = [
  45.055830,
  42.709720,
  36.145221,
  36.053900,
  35.454545,
  35.443578]

lon = [
  142.107145,
  141.565796,
  137.423483,
  140.026900,
  138.762249,
  138.764704]

llon = 127.01666666666667
llat = 30
rlon = 156.64791666666667
ulat = 40

fig = plt.figure(1, figsize=(10, 10))
plt.rcParams['axes.linewidth'] = 2

#axes1 = fig.add_axes([0, 0, 1, 1])
axes1 = fig.add_subplot(111)

Map1 = Basemap(projection='cyl', resolution='i', llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=rlon, urcrnrlat=ulat)

#iamge_data = np.flipud(cv2.imread('test.png')[:, :, [2, 1, 0]])
Map1.imshow(image_data)

Map1.drawcoastlines(color='white', linewidth=0.25)

Map1.drawmeridians(np.arange(llon, rlon+0.1, 0.1), labels=[1, 1, 1, 1], dashes=[1, 2], linewidth=2, color='white')
Map1.drawparallels(np.arange(llat, ulat+0.1, 0.1), labels=[1, 1, 1, 1], dashes=[1, 2], linewidth=2, color='white')

Map1.drawcoastlines(color='white', linewidth=0.25)
Map1.drawcountries(color='white', linewidth=0.25)

axes1.plot(lon, lat, markersize=20, linewidth=0, color='fuchsia', marker='o')

axes1.set_xlim(138.764704-50*d, 138.764704+50*d)
axes1.set_ylim(35.443578-50*d, 35.443578+50*d) 

plt.show()
