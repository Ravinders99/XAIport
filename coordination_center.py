import yaml
import concurrent.futures
import os
import DataProcess
import Model_ResNet
import xai_resnet
import evaluation

def run_task_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Data Processing
    data_processor = DataProcess.DataProcess()
    image_folder = config['data_processing']['image_folder']
    dataset_id = config['data_processing']['dataset_id']
    storage_address = config['data_processing']['storage_address']

    image_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    data_processor.upload_dataset(image_files, dataset_id, storage_address, 'image')
    info = data_processor.get_dataset_info(dataset_id)
    print(info)

    perturbed_dataset_path = data_processor.apply_image_perturbation(
        dataset_id, data_processor.gaussian_noise, config['perturbation']['severity']
    )

    dataset_path0 = info['storage_address']
    dataset_paths = [dataset_path0, perturbed_dataset_path]

    # Model
    Model_ResNet.model_run(dataset_paths)

    # XAI
    xai_resnet.xai_run(dataset_paths)

    # Evaluation
    original_dataset_path = dataset_paths[0]
    model_name = config['model']['name']
    attack_name = "Original"  # Replace with your actual attack name
    CAM_ALGORITHMS = config['xai']['algorithms']

    rs_dir = os.path.join("results", original_dataset_path, model_name)
    evaluation.cam_summary(rs_dir, model_name, attack_name, CAM_ALGORITHMS)

def run_all_tasks_in_parallel():
    yaml_files = [f for f in os.listdir('task_sheets') if f.endswith('.yaml')]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_task_from_yaml, os.path.join('task_sheets', file)) for file in yaml_files]
        for future in concurrent.futures.as_completed(futures):
            print(f"Task {future} completed")

if __name__ == '__main__':
    run_all_tasks_in_parallel()
