from dataclasses import dataclass
import torch


@dataclass
class MelSpectrogramConfig:
    num_mels = 80


@dataclass
class FastSpeech2Config:
    vocab_size = 300
    max_seq_len = 3000

    pitch_vocab = 256
    energy_vocab = 256
    min_pitch = 0
    max_pitch = 861.06526801
    min_energy = 3.17629938e-02
    max_energy = 4.87600768e+02

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernels = [9, 1]

    feature_predictor_filter_size = 256
    feature_predictor_kernel_size = 3
    dropout = 0.1
    
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = ''
    UNK_WORD = ''
    BOS_WORD = ''
    EOS_WORD = ''


@dataclass
class TrainConfig:
    checkpoint_path = "./data/model_new"
    logger_path = "./data/logger"
    mel_ground_truth = "./data/mels"
    alignment_path = "./data/alignments"
    data_path = './data/train.txt'
    pitch_path = './data/pitch'
    energy_path = './data/energy'
    wav_path = './data/LJSpeech-1.1/wavs'
    
    wandb_project = 'TTS'
    wandb_entity = 'broccoliman'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 5000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32

    test_audio = [
        "One ring to rule them all, one ring to find them, one ring to bring them all and in the darkness bind them.",
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest.",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education.",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space."
    ]
