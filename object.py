import open3d
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def getobjects(meta,depth,label):
    intrinsic = meta['intrinsic']
    objects = meta['object_ids']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    objects_list = []
    objects = objects.numpy()
    for object in objects:
        mask = label == object

        objects_list.append(points_viewer[mask].reshape([-1, 3]))

    return objects_list


# depth.png and label.png are in RGB frame
# have extrinsic for sim and real and check openCV for warping
depth = np.array(Image.open('126.png')) / 1000
meta = load_pickle('meta126.pkl')
image = Image.open('126label.png')
new_image = image.resize((960,540))
label = np.array(new_image)

# objects = getobjects(meta, depth,label)
#
# points = open3d.utility.Vector3dVector(objects[1].reshape([-1, 3]))
#
# pcd = open3d.geometry.PointCloud()
# pcd.points = points
#
# open3d.visualization.draw_geometries([pcd])

plt.imshow(depth)
plt.show()
