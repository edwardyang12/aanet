import open3d
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import utils

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

depth = np.array(Image.open('126gt_new.png'))/ 1000

meta = load_pickle('meta126.pkl')

intrinsic = meta['intrinsic']
z = depth

v, u = np.indices(z.shape)
uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
print(np.unique(depth))
mask_depth = (depth > 0.) & (depth < 2)

points_viewer = points_viewer[mask_depth]
print(np.unique(points_viewer))
print(points_viewer.shape)

# points = open3d.utility.Vector3dVector(points_viewer.reshape([-1, 3]))
#
# pcd = open3d.geometry.PointCloud()
# pcd.points = points
#
# open3d.visualization.draw_geometries([pcd])

plt.imshow(depth)
plt.show()
