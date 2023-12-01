from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import Model_ResNet

app = FastAPI()

class DatasetPaths(BaseModel):
    dataset_id: str  # Add dataset_id field to the request body

def run_model_async(dataset_id, model_func):
    # Use dataset_id as needed in your model_func
    dataset_paths = f"dataprocess/datasets/{dataset_id}"
    model_func(dataset_paths)

@app.post("/resnet/{dataset_id}/{perturbation_func_name}/{severity}")
async def run_model1_background(
    dataset_id: str,
    perturbation_func_name: str,
    severity: int,
    background_tasks: BackgroundTasks
):
    # Validate severity level
    if severity < 1 or severity > 5:
        raise HTTPException(status_code=400, detail="Severity level must be between 1 and 5")

    dataset_dir, perturbed_dataset_path = data_processor.get_dataset_paths(dataset_id, perturbation_func_name, severity)

    background_tasks.add_task(Model_ResNet.model_run, dataset_dir, perturbed_dataset_path)
    
    return {
        "message": f"ResNet run for dataset {dataset_id}, perturbation {perturbation_func_name}, and severity {severity} has started in the background"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
