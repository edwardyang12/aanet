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

    for object in objects:
        mask = label == object

        objects_list.append(points_viewer[mask].reshape([-1, 3]))

    return objects_list

depth = np.array(Image.open('126.png')) / 1000
meta = load_pickle('meta126.pkl')
image = Image.open('label126.png')
new_image = image.resize((960,540))
label = np.array(new_image)
