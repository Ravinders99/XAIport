'''
Version 0.0.1
This is the previous try to create a server that can handle the pipeline of the XAI service.
Without the status check system, we can't correctly run the whole pipeline.
This must be reconstructed to a new version.
'''

from fastapi import FastAPI, HTTPException, Request
import httpx
import json
import os
import asyncio
import logging

import aiohttp

app = FastAPI(title="Coordination Center")


async def async_http_post(url, json_data=None, files=None):
    async with httpx.AsyncClient(timeout=120.0) as client:  # Timeout set to 60 seconds
        if json_data:
            response = await client.post(url, json=json_data)
        elif files:
            response = await client.post(url, files=files)
        else:
            response = await client.post(url)

        # 检查是否是307重定向响应
        if response.status_code == 307:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                print(f"Redirecting to {redirect_url}")
                return await async_http_post(redirect_url, json_data, files)

        if response.status_code != 200:
            print(f"Error response: {response.text}")  # 打印出错误响应内容
            logging.error(f"Error in POST to {url}: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()


# 处理上传配置
async def process_upload_config(upload_config):
    # for dataset_id, dataset_info in upload_config['datasets'].items():
    #     url = upload_config['server_url'] + f"/upload-dataset/{dataset_id}"
    #     local_zip_file_path = dataset_info['local_zip_path']  # 每个数据集的本地 ZIP 文件路径

    #     async with httpx.AsyncClient() as client:
    #         with open(local_zip_file_path, 'rb') as f:
    #             files = {'zip_file': (os.path.basename(local_zip_file_path), f)}
    #             response = await client.post(url, files=files)
    #         response.raise_for_status()

    #     # 可以添加更多的逻辑处理上传后的结果
    #     print(f"Uploaded dataset {dataset_id} successfully.")
    for dataset_id, dataset_info in upload_config['datasets'].items():
            local_video_dir = dataset_info.get('local_video_dir')

            if not local_video_dir or not os.path.exists(local_video_dir):
                raise HTTPException(status_code=400, detail=f"Video directory {local_video_dir} not found.")

            # Use the endpoint /process-kinetics-dataset to process the video directory
            url = upload_config['server_url'] + "/process-kinetics-dataset"
            json_data = {"video_dir": local_video_dir, "num_frames": 8}

            # Trigger video processing
            await async_http_post(url, json_data=json_data)
            print(f"Processed video dataset {dataset_id} successfully.")



# 处理扰动配置
async def process_perturbation_config(perturbation_config):
    if not perturbation_config['datasets']:
        print("No perturbation configured, skipping this step.")
        return
    url = perturbation_config['server_url']
    for dataset, settings in perturbation_config['datasets'].items():

        if settings['perturbation_type'] == "none":        #skip perturbation if perturbation_type is none
            print(f"Skipping perturbation for dataset {dataset}.")
            continue
        full_url = f"{url}/apply-perturbation/{dataset}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)


# 处理模型配置
async def process_model_config(model_config):
    base_url = model_config['base_url']
    # for model, settings in model_config['models'].items():
    #     # full_url = f"{base_url}/{settings['model_name']}/{model}/{settings['perturbation_type']}/{settings['severity']}"
    #     full_url = f"{base_url}/{settings['model_name']}/{model}"
    #     print(f"Calling model server: {full_url}")
    #     await async_http_post(full_url)
    # for model, settings in model_config['models'].items():
    #     video_input_dir = settings.get('video_input_dir', 'datasets/FGSM')  # Set FGSM as default

    #     full_url = f"{base_url}/{settings['model_name']}/{model}"
        
    #     data = {
    #         "video_directory": video_input_dir,
    #         "num_frames": settings.get('num_frames', 8)
    #     }
        
    #     print(f"Sending adversarial videos to model server: {video_input_dir}")
    #     await async_http_post(full_url, json_data=data)


    # for model, settings in model_config['models'].items():
    #     original_video_dir = settings.get('original_video_dir', 'dataprocess/videos')  
    #     adversarial_video_dir = settings.get('adversarial_video_dir', 'dataprocess/FGSM')  
        
    #     # Process original videos
    #     full_url_original = f"{base_url}/{settings['model_name']}/{model}"
    #     data_original = {
    #         "video_directory": original_video_dir,
    #         "num_frames": settings.get('num_frames', 8)
    #     }
    #     print(f"Sending original videos to model server: {original_video_dir}")
    #     await async_http_post(full_url_original, json_data=data_original)
        
    #     # Process adversarial videos
    #     full_url_adversarial = f"{base_url}/{settings['model_name']}/{model}"
    #     data_adversarial = {
    #         "video_directory": adversarial_video_dir,
    #         "num_frames": settings.get('num_frames', 8)
    #     }
    #     print(f"Sending adversarial videos to model server: {adversarial_video_dir}")
    #     await async_http_post(full_url_adversarial, json_data=data_adversarial)

    for model, settings in model_config['models'].items():
        original_video_dir = settings.get('original_video_dir', 'dataprocess/test_video')  
        adversarial_video_dir = settings.get('adversarial_video_dir', 'dataprocess/FGSM')  

        # Check if directories exist
        if not os.path.exists(original_video_dir):
            print(f"⚠️ Warning: Original video directory '{original_video_dir}' not found. Skipping.")
            continue

        if not os.path.exists(adversarial_video_dir):
            print(f"⚠️ Warning: Adversarial video directory '{adversarial_video_dir}' not found. Skipping.")

        full_url = f"{base_url}/process-videos"
        print(f"🔄 Sending request to process all videos at: {full_url}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(full_url) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        print("✅ Video processing completed:", json_response)
                    else:
                        print(f"❌ Error processing videos: {response.status}, {await response.text()}")
            except Exception as e:
                print(f"🔥 Exception during request: {e}")
# 处理 XAI 配置
async def process_xai_config(xai_config):
    base_url = xai_config['base_url']
    # for dataset, settings in xai_config['datasets'].items():
        # dataset_id = settings.get('dataset_id', '')  # 提取 "dataset_id"
        # algorithms = settings.get('algorithms', [])  # 提取 "algorithms"
        # data = {
        #     "dataset_id": dataset_id,
        #     "algorithms": algorithms
        # }
        # print(data)
        # full_url = f"{base_url}/cam_xai/"
        # print(full_url)
        # await async_http_post(full_url, json_data=data)
    
    for dataset, settings in xai_config['datasets'].items():
        video_dir = settings.get('video_path', '')
        num_frames = settings.get('num_frames', 8)
        
        if os.path.isdir(video_dir):
            video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            for video_file in video_files:
                data = {
                    "video_path": video_file,
                    "num_frames": num_frames
                }
                full_url = f"{base_url}/staa-video-explain/"
                try:
                    response = await async_http_post(full_url, json_data=data)
                    print(f"XAI response for video {video_file}: {response}")
                except Exception as e:
                    print(f"Error processing XAI for video {video_file}: {e}")
        else:
            print(f"Video path {video_dir} is not a directory.")
# 处理评估配置
async def process_evaluation_config(evaluation_config):
    base_url = evaluation_config['base_url']
    for dataset, settings in evaluation_config['datasets'].items():
        data = {
            "dataset_id": dataset,
            "model_name": settings['model_name'],
            "perturbation_func": settings['perturbation_func'],
            "severity": settings['severity'],
            "cam_algorithms": settings['algorithms']
        }
        full_url = f"{base_url}/evaluate_cam"
        await async_http_post(full_url, json_data=data)

# 按顺序处理每个配置步骤
async def process_pipeline_step(config, step_key, process_function):
    if step_key in config:
        await process_function(config[step_key])

# 从配置运行整个 Pipeline
async def run_pipeline_from_config(config):
    await process_pipeline_step(config, 'upload_config', process_upload_config)
    await process_pipeline_step(config, 'perturbation_config', process_perturbation_config)
    await process_pipeline_step(config, 'model_config', process_model_config)
    await process_pipeline_step(config, 'xai_config', process_xai_config)
    await process_pipeline_step(config, 'evaluation_config', process_evaluation_config)
import traceback  
# API 端点来触发 Pipeline
@app.post("/run_pipeline/")
async def run_pipeline(request: Request):
    config = await request.json()  # 直接从请求中读取 JSON 配置
    try:
        await run_pipeline_from_config(config)
        return {"message": "Pipeline executed successfully"}
    except Exception as e:
        error_details = traceback.format_exc()   # get detailed error message
        print(f"Error encountered in pipeline:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

# 加载配置文件
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)

# import asyncio
# import json

# 假设您的处理函数和其他必要的导入已经完成

# # 加载配置文件
# def load_config():
#     with open("/home/z/Music/devnew_xaiservice/XAIport/task_sheets/task.json", "r") as file:
#         return json.load(file)



# # 主函数
# def main():
#     config = load_config()
#     asyncio.run(run_pipeline_from_config(config))

# if __name__ == "__main__":
#     main()


