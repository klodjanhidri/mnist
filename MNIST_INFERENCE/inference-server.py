import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from MNIST_MODEL.Net import Net
from colorama import Fore
import signal
#import uvicorn
import os
from threading import Thread, Lock
import  time

app = FastAPI()
MODEL_PATH='/workspace/model.pth'
pod_ip = os.environ.get("POD_IP")

class InputData(BaseModel):
    # Define the input data schema
    data: List[List[List[List[float]]]]

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
model_lock = Lock()  # Lock to ensure thread-safe updates
last_mod_time = os.path.getmtime(MODEL_PATH)

if device=='cuda':
    print(Fore.GREEN + "INFO:\t", Fore.WHITE + " Execute MNIST Inference to [", Fore.MAGENTA+"NVIDIA GRAPHICS CARD",Fore.WHITE+"]")
else:
    print(Fore.GREEN + "INFO:\t", Fore.WHITE + " Execute MNIST Inference to [", Fore.MAGENTA+"CPU",Fore.WHITE+"]")


@app.get("/")
async def read_root():
    global test
    test = 'Update Test on POD req'
    print('POD_ID=', pod_ip)
    return {"pod_ip": pod_ip}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/readiness")
async def readiness():
    """
    Monitors the weight file and reloads the model if the file is modified.
    :param path: Path to the weights file.
    """
    global last_mod_time
    print(f"Monitoring weights file: {MODEL_PATH}")
    mod_time = os.path.getmtime(MODEL_PATH)
    if mod_time != last_mod_time:
            print("Detected changes in model weights. Reloading model...")
            global model
            with model_lock:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device) ))  # Reload the model with new weights
            last_mod_time = mod_time
            print("Model weights reloaded successfully.")
    return {"status": "ready"}

@app.post("/predict/")
async def predict(data: InputData):
    result = []
    # Perform inference
    with model_lock:
     model.eval()
     with torch.no_grad():
        for data1 in data.data :
            input_tensor = torch.tensor(data1).to(device)
            return_classes = model(input_tensor)
            predicted_class =  return_classes.argmax(dim=1, keepdim=True)
            # Process output as needed
            result.append(predicted_class.item())
    return {"result": result}

def handle_sigterm(*args):
    print("Received SIGTERM, shutting down gracefully")
    sys.exit(0)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app='inference-server:app', port=8000, workers=1, host='0.0.0.0')


