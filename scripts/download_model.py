import wandb
import os

api = wandb.Api()

os.makedirs("data/download", exist_ok=True)
run = api.run("broccoliman/TTS/3gr8e164")
run.file("data/cool_model/checkpoint_140000.pth.tar").download(root="data/download")
