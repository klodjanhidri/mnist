import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from MNIST_MODEL.Net import Net
from colorama import Fore
import signal 
import uvicorn
import os

app = FastAPI()
MODEL_PATH='/mnt/vol0/minikube/MNIST_MODEL/argo-workflow-bucket/model.pth'

pod_ip = os.environ.get("POD_IP")

@app.get("/")
async def read_root():
    print('POD_ID=', pod_ip)
    return {"pod_ip": pod_ip}


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
model.eval()

if device=='cuda':
    print(Fore.GREEN + "INFO:\t", Fore.WHITE + " Execute MNIST Inference to [", Fore.MAGENTA+"NVIDIA GRAPHICS CARD",Fore.WHITE+"]")
else:     
    print(Fore.GREEN + "INFO:\t", Fore.WHITE + " Execute MNIST Inference to [", Fore.MAGENTA+"CPU",Fore.WHITE+"]")


@app.post("/predict/")
async def predict(data: InputData):
    result = []  
    # Perform inference
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
    uvicorn.run(app='inference-server:app', port=8000, host='0.0.0.0')
