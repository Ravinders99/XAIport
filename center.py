import os
from DataProcess import DataProcess

# # 实例化 DataProcess 类
data_processor = DataProcess()

# 获取图片文件夹中的所有图片路径
image_folder = 'XAIport/n01440764'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 上传数据集
dataset_id = 'test_dataset'
storage_address = 'XAIport/data'  # 指定存储地址
data_processor.upload_dataset(image_files, dataset_id, storage_address, 'image')

# 获取并打印数据集信息
info = data_processor.get_dataset_info(dataset_id)
print(info)

# 你可以在这里继续添加对其他方法的测试，比如 update_dataset, delete_dataset 等


data_processor.apply_image_perturbation('test_dataset', data_processor.gaussian_noise, 3) 