import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        # www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        # updated by Xu for new dataset link;
        # the file is from: https://huggingface.co/datasets/Msun/modelnet40/tree/main
        www = "https://github.com/ma-xu/pointMLP-pytorch/releases/download/Modenet40_dataset/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    mn40_sort = [
        8, 30, 0, 4, 2, 37, 22, 33, 35, 5,
        21, 36, 26, 25, 12, 23, 14, 7, 3, 16,
        9, 34, 17, 15, 20, 18, 11, 29, 31, 19,
        28, 13, 1, 27, 39, 32, 24, 38, 10, 6]
    task_data_label = {}
    for i in range(9):
        if i == 0:
            task = mn40_sort[:16]
        else:
            task = mn40_sort[13 + i * 3: 16 + i * 3]
        mask = np.isin(all_label, list(task))
        idx = mask.nonzero()[0]
        task_sample = all_data[idx]
        task_label = all_label[idx]
        # print('label', task_label.shape)
        label_indices = task_label.flatten()
        label_indices_store = {}
        for label in task:
            indices = np.where(label_indices == label)[0]
            label_indices_store[label] = indices
        task_data_label[i] = {
            'i': i, 'task': task, 'data': task_sample, 'label': task_label, 'label_indices': label_indices_store
        }
    return task_data_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class CILSampler(Sampler):
    def __init__(self, dataset, indices, episode, way, shot, query):
        super(CILSampler, self).__init__(data_source=dataset)
        self.dataset = dataset
        self.indices = indices
        self.episode = episode
        self.way = way
        self.shot = shot
        self.query = query

        self.data = self.dataset.data
        self.label = self.dataset.label

    def __iter__(self):
        for _ in range(self.episode):
            class_list = list(set([item for sublist in self.label for item in sublist]))
            selected_classes = np.random.choice(class_list, self.way, replace=False)
            support_set = []
            query_set = []
            for c in selected_classes:
                class_indices = self.indices[c]
                selected_indices = np.random.choice(class_indices, self.shot + self.query, replace=False)
                support_set.extend(selected_indices[:self.shot])
                query_set.extend(selected_indices[self.shot:])

            yield support_set + query_set

    def __len__(self):
        return self.episode


class ModelNet40_CIL(Dataset):
    def __init__(self, data, label, num_points, partition='train'):
        self.data, self.label = data, label
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def get_dataset_complete():
    train_dataset = load_data('train')
    val_dataset = load_data('test')
    return train_dataset, val_dataset


def get_sets(train_task, val_task, way=5, shot=1, query=15):
    train_data, train_label, train_label_indices = train_task['data'], train_task['label'], train_task['label_indices']
    train_dataset = ModelNet40_CIL(partition='train', num_points=1024, data=train_data, label=train_label)
    train_sampler = CILSampler(dataset=train_dataset, indices=train_label_indices, episode=500, way=way, shot=shot, query=query)
    train_loader = DataLoader(train_dataset, num_workers=4, batch_sampler=train_sampler)

    val_data, val_label, val_label_indices = val_task['data'], val_task['label'], val_task['label_indices']
    val_dataset = ModelNet40_CIL(partition='test', num_points=1024, data=val_data, label=val_label)
    val_sampler = CILSampler(dataset=val_dataset, indices=val_label_indices, episode=500, way=way, shot=shot, query=query)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_sampler=val_sampler)

    return train_loader, val_loader

