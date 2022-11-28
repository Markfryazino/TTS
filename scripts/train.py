from src.configs import TrainConfig, FastSpeech2Config, MelSpectrogramConfig
from src.training.training_loop import train, prepare

model_config = FastSpeech2Config()
train_config = TrainConfig()
mel_config = MelSpectrogramConfig()

model, train_loader, fastspeech_loss, optimizer, scheduler, wave_glow = prepare(model_config, train_config, mel_config)
train(train_config, model, train_loader, fastspeech_loss, optimizer, scheduler, wave_glow)
