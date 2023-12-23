#%%
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib import cm

#path = 'ground-truth/data_viz/16k/'
path = 'readable/AP/16k/'



file_list = []
for filename in os.listdir(path):
    file_list.append(filename)
print('number of files: ' + str(len(filename)))
filename = file_list[32] #10, 22, 32 spiral, 
filename = path + filename

data = np.load(filename)
for key in data.keys():
    array = data[key]
    print(f"Data for key {key}:")
    print(array)
xyz = data['XYZ']
voltage = data['Voltage']
print('number of points: ' + str(len(voltage)))
print('check: ' + str(len(voltage)) + ' = ' + str(len(xyz))) 
print(np.max(voltage))



grayscale_colors = np.squeeze(voltage)*0.9+0.05
color_mapped = cm.magma(grayscale_colors)
colors = np.zeros((len(voltage), 3))
colors[:, 0] = color_mapped[:,0]
colors[:, 1] = color_mapped[:,1]
colors[:, 2] = color_mapped[:,2]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz[::4])
pcd.colors = o3d.utility.Vector3dVector(colors[::4])

points = np.asarray(pcd.points)

# Extract colors
colors = np.asarray(pcd.colors)

# Create a voxel grid
voxel_size = 6.0
volume = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)


# Visualize the voxel grid
o3d.visualization.draw_geometries([pcd])

# %%
