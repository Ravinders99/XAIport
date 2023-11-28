import os
from DataProcess import DataProcess
from Model_ResNet import Model_ResNet

# # 实例化 DataProcess 类
data_processor = DataProcess()

# 获取图片文件夹中的所有图片路径
image_folder = 'n01440764'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 上传数据集
dataset_id = 'test_dataset'
storage_address = 'data'  # 指定存储地址
data_processor.upload_dataset(image_files, dataset_id, storage_address, 'image')

# 获取并打印数据集信息
info = data_processor.get_dataset_info(dataset_id)
print(info)

# 你可以在这里继续添加对其他方法的测试，比如 update_dataset, delete_dataset 等


data_processor.apply_image_perturbation('test_dataset', data_processor.gaussian_noise, 3) 



# 根据 DataProcess 类的结果设置 dataset_path
dataset_path = info['storage_address']
print(dataset_path)
class_index_path = 'index/imagenet_class_index.json'
# model_resnet = Model_ResNet(dataset_path, class_index_path)
# model_resnet.evaluate()