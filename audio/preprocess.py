import librosa, soundfile as sf
from pydub import AudioSegment
from typing import List
from settings import settings

def resample_to_16k_mono(in_path: str, out_path: str):
    y, sr = librosa.load(in_path, sr=None, mono=True)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=settings.SAMPLE_RATE)
    sf.write(out_path, y16, settings.SAMPLE_RATE)

def split_audio_30s(path_16k: str) -> List[str]:
    audio = AudioSegment.from_file(path_16k).set_channels(1).set_frame_rate(settings.SAMPLE_RATE)
    chunk_ms = settings.WINDOW_SECONDS * 1000
    paths = []
    for i in range(0, len(audio), chunk_ms):
        part = audio[i:i+chunk_ms]
        out = f"{path_16k}.{i//chunk_ms:02d}.wav"
        part.export(out, format="wav", parameters=["-ac","1","-ar",str(settings.SAMPLE_RATE)])
        paths.append(out)
    return paths
