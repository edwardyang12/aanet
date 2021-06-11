import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

camera_info = load_pickle('cam_db_full.pkl')
print(camera_info)
