from fastapi import FastAPI, BackgroundTasks, HTTPException ,File

from pydantic import BaseModel
import Model_ResNet
import os
import torch
from attention_extractor import AttentionExtractor  
from fastapi import UploadFile

app = FastAPI()
# Initialize STAA
attention_extractor = AttentionExtractor(model_name="facebook/timesformer-base-finetuned-k400", device="cuda" if torch.cuda.is_available() else "cpu")
# Output directory
OUTPUT_DIR = "video_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DatasetPaths(BaseModel):
    dataset_id: str  # Add dataset_id field to the request body

def run_model_async(dataset_id, model_func):
    # Use dataset_id as needed in your model_func
    dataset_paths = f"dataprocess/datasets/{dataset_id}"
    model_func(dataset_paths)


@app.post("/resnet/{dataset_id}/{perturbation_func_name}/{severity}")
async def run_model1_background(dataset_id: str, perturbation_func_name: str, severity: int, background_tasks: BackgroundTasks):
    local_original_dataset_path = f"/home/z/Music/devnew_xaiservice/XAIport/datasets/{dataset_id}"
    local_perturbed_dataset_path = f"/home/z/Music/devnew_xaiservice/XAIport/datasets/{dataset_id}_{perturbation_func_name}_{severity}"

    # 构建 dataset_paths 列表
    dataset_paths = [local_original_dataset_path, local_perturbed_dataset_path]

    # 异步运行模型
    background_tasks.add_task(Model_ResNet.model_run, dataset_paths)

    return {
        "message": f"ResNet run for dataset {dataset_id} with perturbation {perturbation_func_name} and severity {severity} has started, results will be uploaded to Blob storage after computation."
    }


# video explanation
extractor = AttentionExtractor(
    model_name="facebook/timesformer-base-finetuned-k400",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Directories
TEMP_DIR = "temp"
OUTPUT_DIR = "video_results"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/video-explain/")
async def video_explain(video: UploadFile = File(...)):
    """
    Endpoint to process video and extract attention using the Timesformer model.
    """
    try:
        # Save uploaded video to a temporary file
        video_path = os.path.join(TEMP_DIR, video.filename)
        with open(video_path, "wb") as f:
            f.write(await video.read())

        # Process video with the AttentionExtractor
        spatial_attention, temporal_attention, frames, logits = extractor.extract_attention(video_path)
        prediction_idx = torch.argmax(logits, dim=1).item()
        prediction = extractor.model.config.id2label[prediction_idx]

        # Save visualization results
        save_path = os.path.join(OUTPUT_DIR, os.path.splitext(video.filename)[0])
        os.makedirs(save_path, exist_ok=True)
        extractor.visualize_attention(
            spatial_attention, temporal_attention, frames, save_path, prediction, "Unknown"
        )

        # Cleanup temporary video
        os.remove(video_path)

        return {
            "message": "Video processed successfully.",
            "prediction": prediction,
            "results_dir": save_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)