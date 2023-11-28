import os
import shutil
from PIL import Image
import numpy as np
import random
from PIL import Image, ImageFilter
import numpy as np

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

    def apply_image_perturbation(self, dataset_id, perturbation_func, severity=1):
        """ 对图像数据集应用变换（perturbation）"""
        if dataset_id not in self.datasets:
            raise ValueError("Dataset ID does not exist.")

        if self.dataset_properties[dataset_id]["data_type"] != "image":
            raise ValueError("Perturbation can only be applied to image datasets.")

        dataset_dir = self.dataset_properties[dataset_id]["storage_address"]

        # 新的父文件夹路径
        new_parent_folder_name = f"{dataset_id}_perturbation_{perturbation_func.__name__}_{severity}"
        new_parent_folder_path = os.path.join(os.path.dirname(dataset_dir), new_parent_folder_name)
        os.makedirs(new_parent_folder_path, exist_ok=True)

        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            if os.path.isdir(subdir_path):
                # 创建相应的新子目录
                new_subdir_path = os.path.join(new_parent_folder_path, subdir)
                os.makedirs(new_subdir_path, exist_ok=True)

                # 处理子目录中的每个文件
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path) and 'original' in file:
                        try:
                            image = Image.open(file_path)
                            perturbed_image = perturbation_func(image, severity)

                            # 保存扰动后的图像到新子目录，保持原始文件名
                            perturbed_path = os.path.join(new_subdir_path, file)
                            perturbed_image.save(perturbed_path)
                            print(f"Saved perturbed image to {perturbed_path}")
                        except Exception as e:
                            print(f"Failed to process file: {file_path}, Error: {e}")
