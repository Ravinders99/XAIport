import aiofiles
import os
import DataProcess
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import zipfile
from typing import List
from fastapi import BackgroundTasks
from functionaltool.cloudstorage import up_cloud, down_cloud

app = FastAPI()

data_processor = DataProcess.DataProcess(base_storage_address="datasets")

@app.post("/upload-dataset/{dataset_id}")
async def upload_dataset(dataset_id: str, zip_file: UploadFile = File(...)):
    # Ensure temp directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Save ZIP file
    temp_file = f"{temp_dir}/{zip_file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    # Prepare dataset directory
    dataset_dir = f"datasets/{dataset_id}"
    os.makedirs(dataset_dir, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Skip directories
            if member.is_dir():
                continue

            # Construct target file path without top-level directory
            target_path = os.path.join(dataset_dir, '/'.join(member.filename.split('/')[1:]))

            # Create any intermediate directories
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Extract file to the target path
            with zip_ref.open(member, 'r') as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

    # 在成功解压缩文件后，上传整个数据集文件夹到云存储
    cloud_upload_path = os.path.join("datasets", dataset_id)
    up_cloud(dataset_dir, cloud_upload_path)

    return {"message": "Dataset uploaded to local storage and Azure Blob Storage successfully"}

    # Clean up
    os.remove(temp_file)

    return {"message": "Dataset uploaded and extracted successfully"}



@app.get("/get-dataset-info/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    try:
        info = data_processor.get_dataset_info(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return info




@app.delete("/delete-dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    try:
        data_processor.delete_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": "Dataset deleted successfully"}

@app.get("/download-dataset/{dataset_id}")
async def download_dataset(dataset_id: str, download_path: str):
    try:
        data_processor.download_dataset(dataset_id, download_path)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": "Dataset downloaded successfully"}


@app.post("/apply-perturbation/{dataset_id}/{perturbation_func_name}/{severity}")
async def apply_perturbation(background_tasks: BackgroundTasks, dataset_id: str, perturbation_func_name: str, severity: int):
    # 确认扰动函数存在于 DataProcess 类中且可调用
    if not hasattr(DataProcess.DataProcess, perturbation_func_name) or not callable(getattr(DataProcess.DataProcess, perturbation_func_name)):
        raise HTTPException(status_code=400, detail="Unsupported or invalid perturbation function.")

    # 获取对应的扰动函数
    perturbation_func = getattr(DataProcess.DataProcess, perturbation_func_name)

    # 异步执行扰动应用
    background_tasks.add_task(data_processor.apply_image_perturbation, dataset_id, perturbation_func, severity)

    return {"message": "Perturbation process started."}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

    
# from fastapi.responses import FileResponse
# import zipfile
# import tempfile

# @app.get("/download-dataset/{dataset_id}")
# async def download_dataset_zip(dataset_id: str, perturbation_func_name: str = None, severity: int = None):
#     # 定义原始和扰动数据集的路径
#     original_dataset_path = os.path.join(data_processor.base_storage_address, dataset_id)
#     perturbed_dataset_path = None

#     # 如果提供了扰动函数名和严重程度，则构建扰动数据集的路径
#     if perturbation_func_name and severity is not None:
#         perturbed_folder_name = f"{dataset_id}_perturbation_{perturbation_func_name}_{severity}"
#         perturbed_dataset_path = os.path.join(data_processor.base_storage_address, perturbed_folder_name)

#     # 确认原始文件夹存在
#     if not os.path.exists(original_dataset_path):
#         raise HTTPException(status_code=404, detail="Original dataset not found.")

#     # 创建临时文件用于存储ZIP文件
#     temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

#     with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         # 添加原始数据集到ZIP文件
#         for root, dirs, files in os.walk(original_dataset_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 zipf.write(file_path, file_path[len(data_processor.base_storage_address)+1:])

#         # 如果存在扰动数据集，也添加到ZIP文件
#         if perturbed_dataset_path and os.path.exists(perturbed_dataset_path):
#             for root, dirs, files in os.walk(perturbed_dataset_path):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     zipf.write(file_path, file_path[len(data_processor.base_storage_address)+1:])

#     return FileResponse(temp_zip.name, media_type='application/octet-stream', filename=f"{dataset_id}.zip")
