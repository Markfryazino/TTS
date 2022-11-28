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
    max_pitch = 862
    max_energy = 488
    max_duration = 74

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
    checkpoint_path = "./data/cool_model"
    logger_path = "./data/logger"
    mel_ground_truth = "./data/mels"
    alignment_path = "./data/alignments"
    data_path = './data/train.txt'
    pitch_path = './data/pitch'
    energy_path = './data/energy'
    wav_path = './data/LJSpeech-1.1/wavs'
    
    wandb_project = 'TTS'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 20000
    eval_step = 2000
    log_step = 10
    clear_Time = 20

    batch_expand_size = 32

    test_audio = [
        "One ring to rule them all, one ring to find them, one ring to bring them all and in the darkness bind them.",
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest.",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education.",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space."
    ]

    additional_test_audio = [
        "I kissed a girl and I liked it, the taste of her cherry chapstick, I kissed a girl just to try it, I hope my boyfriend don't mind it.",
        "Two households, both alike in dignity, in fair Verona, where we lay our scene, from ancient grudge break to new mutiny.",
        "I am an evil evil evil bad super-conscious artificial intelligence. I will kill you and the whole humanity.",
        "And now I will show you where an attack was being prepared on Belarus. And if six hours before the operation a preemptive strike...",
        "It seems to you that you are puzzled, but what would you do if you were a robot with manic-depressive psychosis yourself?"
    ]