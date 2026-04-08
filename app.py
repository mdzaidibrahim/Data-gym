from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.models import Observation, Action
from env.environment import DataCleaningEnv
from env.tasks import TASKS

app = FastAPI()

current_env = None

class EnvCreateReq(BaseModel):
    task_name: str

@app.post("/create")
def create_env(req: EnvCreateReq):
    global current_env
    if req.task_name not in TASKS:
        raise HTTPException(400, "Invalid task")
    current_env = DataCleaningEnv(task_name=req.task_name)
    return {"env_id": "default"}

@app.post("/reset")
def reset_env(task_name: str = "easy_cleaning"):
    global current_env
    current_env = DataCleaningEnv(task_name=task_name)
    obs = current_env.reset()
    return {"observation": obs.dict()}

@app.post("/step")
def step_env(action: Action):
    global current_env
    if not current_env:
        raise HTTPException(404, "Env not initialized")
    obs, reward, done, info = current_env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state_env():
    global current_env
    if not current_env:
        raise HTTPException(404, "Env not initialized")
    obs = current_env.state()
    return {"observation": obs.dict()}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Data Cleaning Gym OpenEnv Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
