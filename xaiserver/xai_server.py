from pydantic import BaseModel
from typing import List
import logging
import cam_resnet
from fastapi import FastAPI, BackgroundTasks, HTTPException,Request
import aiohttp
import os
from attention_extractor import AttentionExtractor  
import torch 
import json
import matplotlib.pyplot as plt
app = FastAPI()
staa_model = AttentionExtractor("facebook/timesformer-base-finetuned-k400", device="cpu")
class XAIRequest(BaseModel):
    dataset_id: str
    algorithms: List[str]

async def async_http_post(url, json_data=None, files=None):
    """
    Make asynchronous POST requests to a given URL with JSON data or files.
    """
    async with aiohttp.ClientSession() as session:
        if json_data:
            response = await session.post(url, json=json_data)
        elif files:
            response = await session.post(url, data=files)
        else:
            response = await session.post(url)

        if response.status != 200:
            logging.error(f"Error in POST request to {url}: {response.status} - {await response.text()}")
            raise HTTPException(status_code=response.status, detail=await response.text())

        return await response.json()
async def download_dataset(dataset_id: str) -> str:
    """Download the dataset and return the local dataset path."""
    try:
        local_dataset_path = f"/home/z/Music/devnew_xaiservice/XAIport/datasets/{dataset_id}"
        #down_cloud(f"datasets/{dataset_id}", local_dataset_path)
        return local_dataset_path
    except Exception as e:
        logging.error(f"Error downloading dataset {dataset_id}: {e}")
        raise

async def run_xai_process(dataset_id: str, algorithm_names: List[str]):
    try:
        local_dataset_path = await download_dataset(dataset_id)
        dataset_dirs = [local_dataset_path]

        # 将算法名称转换为算法类
        selected_algorithms = [cam_resnet.CAM_ALGORITHMS_MAPPING[name] for name in algorithm_names]

        cam_resnet.xai_run(dataset_dirs, selected_algorithms)
        # 处理上传结果和其他后续处理
    except Exception as e:
        logging.error(f"Error in run_xai_process: {e}")
        raise


@app.post("/cam_xai")
async def run_xai(request: XAIRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_xai_process, request.dataset_id, request.algorithms)
        return {"message": "XAI processing for dataset has started successfully."}
    except Exception as e:
        logging.error(f"Error in run_xai endpoint: {e}") 
        raise HTTPException(status_code=500, detail=str(e))


extractor = AttentionExtractor("facebook/timesformer-base-finetuned-k400")
class VideoExplainRequest(BaseModel):
    video_path: str
    num_frames: int = 8


@app.post("/staa-video-explain/")
async def staa_video_explain(request: VideoExplainRequest):
    try:
        video_path = request.video_path
        num_frames = request.num_frames
        spatial_attention, temporal_attention, frames, logits = staa_model.extract_attention(video_path, num_frames)
        prediction_idx = torch.argmax(logits, dim=1).item()
        prediction = staa_model.model.config.id2label[prediction_idx]

        return {
            "prediction": prediction,
            "spatial_attention": spatial_attention.tolist(),
            "temporal_attention": temporal_attention.tolist(),
        }
    except Exception as e:
        logging.error(f"Error in staa_video_explain: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
@app.post("/visualize-attention/{video_name}")
async def visualize_attention(video_name: str):
    try:
        # Load the attention JSON file
        json_path = f"video_results/{video_name}/{video_name}_rs.json"
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"Attention data for '{video_name}' not found.")

        with open(json_path, 'r') as f:
            attention_data = json.load(f)

        # Extract and plot temporal attention
        temporal_attention = [frame['mean_attention'] for frame in attention_data]

        # Plot temporal attention
        plt.figure(figsize=(10, 5))
        plt.plot(temporal_attention, label=f'Temporal Attention - {video_name}', color='blue', marker='o')
        plt.xlabel("Frame Number")
        plt.ylabel("Attention Value")
        plt.legend()
        plt.grid(True)
        plt.title(f"Temporal Attention Plot for {video_name}")

        # Save plot
        output_path = f"video_results/{video_name}/{video_name}_temporal_plot.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        return {"message": f"Attention visualization completed for {video_name}", "temporal_plot": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
