import scipy.io as sio
import torch
import os
from torch.utils.data import Dataset

DATASET_TYPES = ["CWRU", "hgd", "PU"]


class BaseDataset(Dataset):
    def __init__(self, root_dir, dataset_type, data_length=1024, stride=1024, num_of_samples=None):
        assert dataset_type in DATASET_TYPES, "数据集类型必须是{}中的一种".format(DATASET_TYPES)
        assert isinstance(data_length, int), "数据的长度必须是整数类型"
        assert isinstance(stride, int), "数据读取的步长必须是整数类型"

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.data_length = data_length
        self.stride = stride
        self.num_of_samples = num_of_samples

        self.data = []
        self.label = []

        self.Load_Data()

    def __getitem__(self, item):
        data = self.data[item]
        data = torch.tensor(data, dtype=torch.float32)
        label = self.label[item]
        label = torch.tensor(label, dtype=torch.float32)
        return data, label

    def __len__(self):
        return len(self.data)

    def Load_Data(self):
        for class_idx, class_folder in enumerate(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if self.dataset_type == "CWRU":
                        batches = self.Load_CWRU(file_path, file_name)
                        self.data.extend(batches)
                        self.label.extend([class_idx] * len(batches))
                    elif self.dataset_type == "hgd":
                        batches = self.Load_hgd(file_path, file_name)
                        self.data.extend(batches)
                        self.label.extend([class_idx] * len(batches))
                    elif self.dataset_type == "PU":
                        batches = self.Load_PU(file_path, file_name)
                        self.data.extend(batches)
                        self.label.extend([class_idx] * len(batches))

    def Load_CWRU(self, file_path, file_name):
        data = sio.loadmat(file_path)
        file_name = file_name.replace(".mat", "")
        formatted_file_name = file_name.zfill(3)
        col_name = f'X{formatted_file_name}_DE_time'
        col_data = data[col_name]
        batches = self.Data_Segmentation(col_data)
        return batches

    def Load_hgd(self, file_path, file_name):
        data = sio.loadmat(file_path)
        col_name = 'Data'
        col_data = data[col_name]
        col_data = col_data[:, 3]
        batches = self.Data_Segmentation(col_data)
        return batches

    def Load_PU(self, file_path, file_name):
        return NotImplementedError("Functions not yet implemented")

    def Data_Segmentation(self, col_data):
        batches = []
        num_segments = ((
                                    len(col_data) - self.data_length) // self.stride) if self.num_of_samples is not None else self.num_of_samples
        assert (num_segments * self.data_length) < len(col_data), "数据长度不足以每个文件生成{}个样本".format(
            self.num_of_samples)
        for i in range(num_segments):
            segment = col_data[i * self.stride: (i * self.stride + self.data_length)]
            batches.append(segment)
        return batches


if __name__ == '__main__':
    root_dir = "./CWRU/bearing"
    dataset_type = "CWRU"
    data_length = 1024
    stride = 1024
    num_of_samples = None
    DATASET = BaseDataset(root_dir, dataset_type, data_length, stride, num_of_samples)
