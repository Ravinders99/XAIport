import os
import shutil
from PIL import Image
import numpy as np
import random

class DataProcess:
    def __init__(self):
        self.datasets = {}
        self.dataset_properties = {}

    def upload_dataset(self, data_files, dataset_id, base_storage_address, data_type):
        """ 上传数据集 """
        if dataset_id in self.datasets:
            raise ValueError("Dataset ID already exists.")

        # 构建数据集的完整存储路径
        dataset_dir = os.path.join(base_storage_address, data_type, dataset_id)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        for file in data_files:
            file_name = os.path.splitext(os.path.basename(file))[0]  # 获取不带扩展名的文件名
            file_extension = os.path.splitext(file)[1]  # 获取文件扩展名
            file_folder = os.path.join(dataset_dir, file_name)
            os.makedirs(file_folder, exist_ok=True)

            # 将文件复制到新位置并保留原始扩展名
            dest_file_name = 'original' + file_extension
            shutil.copy(file, os.path.join(file_folder, dest_file_name))

        # 更新类属性
        self.datasets[dataset_id] = data_files
        self.dataset_properties[dataset_id] = {
            "storage_address": dataset_dir,
            "data_type": data_type,
            "num_files": len(data_files)
        }

    def get_dataset_info(self, dataset_id):
        """ 获取数据集的信息 """
        if dataset_id not in self.datasets:
            raise ValueError("Dataset ID does not exist.")
        return self.dataset_properties[dataset_id]

    def update_dataset(self, data_files, dataset_id):
        """ 更新数据集，添加新的数据 """
        if dataset_id not in self.datasets:
            raise ValueError("Dataset ID does not exist.")
        
        dataset_dir = self.dataset_properties[dataset_id]["storage_address"]
        for file in data_files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            file_folder = os.path.join(dataset_dir, file_name)
            os.makedirs(file_folder, exist_ok=True)
            shutil.copy(file, os.path.join(file_folder, 'original'))

        self.datasets[dataset_id].extend(data_files)
        self.dataset_properties[dataset_id]["num_files"] = len(self.datasets[dataset_id])

    def delete_dataset(self, dataset_id):
        """ 删除整个数据集 """
        if dataset_id not in self.datasets:
            raise ValueError("Dataset ID does not exist.")
        
        dataset_dir = self.dataset_properties[dataset_id]["storage_address"]
        shutil.rmtree(dataset_dir)
        del self.datasets[dataset_id]
        del self.dataset_properties[dataset_id]

    def download_dataset(self, dataset_id, download_path):
        """ 下载整个数据集 """
        if dataset_id not in self.datasets:
            raise ValueError("Dataset ID does not exist.")

        dataset_dir = self.dataset_properties[dataset_id]["storage_address"]
        shutil.copytree(dataset_dir, download_path)

    def gaussian_noise(self, image, severity=1):
        """ 对图像添加高斯噪声 """
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        pil_image = np.array(image) / 255.0
        noisy_image = np.clip(pil_image + np.random.normal(size=pil_image.shape, scale=c), 0, 1) * 255
        return Image.fromarray(noisy_image.astype(np.uint8))

    def blur(self, image, severity=1):
        """ 对图像应用模糊效果 """
        from PIL import ImageFilter
        c = [ImageFilter.BLUR, ImageFilter.GaussianBlur(2), ImageFilter.GaussianBlur(3), ImageFilter.GaussianBlur(5), ImageFilter.GaussianBlur(7)][severity - 1]
        return image.filter(c)


    def apply_image_perturbation(self, dataset_id, perturbation_func):
        """ 对图像数据集应用变换（perturbation）"""
        if dataset_id not in self.datasets:
            raise ValueError("Dataset ID does not exist.")

        if self.dataset_properties[dataset_id]["data_type"] != "image":
            raise ValueError("Perturbation can only be applied to image datasets.")

        dataset_dir = self.dataset_properties[dataset_id]["storage_address"]

        for file_name in os.listdir(dataset_dir):
            file_dir = os.path.join(dataset_dir, file_name)
            original_file_path = os.path.join(file_dir, 'original')
            if os.path.isfile(original_file_path):
                try:
                    image = Image.open(original_file_path)
                    perturbed_image = perturbation_func(image)
                    perturbed_path = os.path.join(file_dir, 'perturbed.png')
                    perturbed_image.save(perturbed_path)
                except IOError:
                    print(f"Failed to process file: {original_file_path}")



def gaussian_noise(image):
