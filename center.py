import os
from DataProcess import DataProcess

# # 实例化 DataProcess 类
# data_processor = DataProcess()

# # 获取图片文件夹中的所有图片路径
# image_folder = 'val_images10k'
# image_files = []
# for root, dirs, files in os.walk(image_folder):
#     for file in files:
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_files.append(os.path.join(root, file))

# # 上传数据集
# dataset_id = 'test_dataset'
# storage_address = 'data'  # 指定存储地址
# data_processor.upload_dataset(image_files, dataset_id, storage_address, 'image')

# # 获取并打印数据集信息
# info = data_processor.get_dataset_info(dataset_id)
# print(info)

# # 应用图像扰动
# data_processor.apply_image_perturbation('test_dataset', data_processor.gaussian_noise, 3)

# # 根据 DataProcess 类的结果设置 dataset_path
# dataset_path = info['storage_address']
# print(dataset_path)

############################################################################################################
# from Model_ResNet import resnet_run
dataset_paths = [
    "data/image/test_dataset",
    "data/image/test_dataset_perturbation_gaussian_noise_3"
]

# resnet_run(dataset_paths)


from xai import resnet_xai_run

resnet_xai_run(dataset_paths)