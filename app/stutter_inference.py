import torch
import torchaudio
import torch.nn.functional as F

def audio_to_mel_64x118(
    file_path,
    target_sr=16000,
    n_mels=64,
    target_frames=118
):
    waveform, sr = torchaudio.load(file_path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels
    )
    mel = mel_transform(waveform)

    mel = torchaudio.functional.amplitude_to_DB(
        mel,
        multiplier=10,
        amin=1e-10,
        db_multiplier=0,
    )

    mel = mel.squeeze(0)  

    # Pad or crop to fixed width
    T = mel.size(1)
    if T < target_frames:
        mel = F.pad(mel, (0, target_frames - T))
    else:
        mel = mel[:, :target_frames]

    return mel


def load_saved_model(model_class, checkpoint_path, device="cpu"):
    model = model_class()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_from_audio(model, audio_path, device="cpu", threshold=0.5):
    mel = audio_to_mel_64x118(audio_path)
    mel = mel.unsqueeze(0).to(device)  # (1, 64, 118)

    with torch.no_grad():
        logits = model(mel)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
    return probs.cpu()[0], preds.cpu()[0]


class StutterPredictor:
    def __init__(self, model_class, checkpoint_path, device="cpu"):
        self.device = device
        self.model = load_saved_model(model_class, checkpoint_path, device)

    def predict(self, audio_file, threshold=0.5):
        return predict_from_audio(
            self.model,
            audio_file,
            device=self.device,
            threshold=threshold
        )
