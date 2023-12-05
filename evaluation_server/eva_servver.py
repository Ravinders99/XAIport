from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import evaluation

app = FastAPI()

# 使用 Field 来提供默认值为 None
class EvaluationRequest(BaseModel):
    dataset_id: str
    model_name: str
    perturbation_func: str = Field(default=None)
    severity: str = Field(default=None)
    cam_algorithms: list

def run_evaluation(dataset_id: str, model_name: str, perturbation_func: str, severity: str, cam_algorithms: list):
    try:
        # 判断 perturbation_func 和 severity 是否被提供
        if perturbation_func and severity:
            attack_name = f"{perturbation_func}_{severity}"
            rs_dir = f"/home/z/Music/devnew_xaiservice/XAIport/xairesult/{dataset_id}_{perturbation_func}_{severity}/{model_name}"
        else:
            attack_name = "Original"  # 或其他默认攻击名称
            rs_dir = f"/home/z/Music/devnew_xaiservice/XAIport/xairesult/{dataset_id}/{model_name}"

        # 调用 cam_summary，传递正确的参数
        evaluation.cam_summary(rs_dir, model_name, attack_name, cam_algorithms)
    except Exception as e:
        print(f"Error in running evaluation: {e}")
        raise


@app.post("/evaluate_cam")
async def evaluate_cam(request: EvaluationRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_evaluation, request.dataset_id, request.model_name, request.perturbation_func, request.severity, request.cam_algorithms)
        return {"message": "CAM evaluation has started successfully."}
    except Exception as e:
        print(f"Error in evaluate_cam endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
