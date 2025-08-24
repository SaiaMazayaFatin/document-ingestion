import librosa, soundfile as sf
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from settings import settings
import math, os, shutil
try:
    from pydub import AudioSegment  # type: ignore
except Exception:  # pydub optional (fallback ke pure librosa splitting)
    AudioSegment = None  # type: ignore

def resample_to_16k_mono(in_path: str, out_path: str):
    """Resample file ke 16k mono (librosa)"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y, sr = librosa.load(in_path, sr=None, mono=True)
    if sr != settings.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=settings.SAMPLE_RATE)
    sf.write(out_path, y, settings.SAMPLE_RATE)

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None


def split_audio_with_overlap(path_16k: str) -> List[str]:
    """Potong audio 16k mono menjadi window + overlap.

    Fallback otomatis ke pemotongan berbasis numpy+librosa jika ffmpeg/pydub tidak tersedia.
    Output: stored in data/interim/audio_segments/ dengan naming clean
    """
    win_s = settings.WINDOW_SECONDS
    ov_s = settings.AUDIO_OVERLAP_SECONDS
    sr = settings.SAMPLE_RATE

    # Create segments directory
    base_name = os.path.splitext(os.path.basename(path_16k))[0]
    segments_dir = "data/interim/audio_segments"
    os.makedirs(segments_dir, exist_ok=True)

    # Prefer pydub+ffmpeg jika tersedia (lebih cepat untuk file besar tanpa memuat seluruh waveform ke RAM sekali lagi)
    if AudioSegment is not None and _ffmpeg_available():
        audio = AudioSegment.from_file(path_16k).set_channels(1).set_frame_rate(sr)
        win_ms = win_s * 1000
        ov_ms = ov_s * 1000
        step = win_ms - ov_ms if win_ms > ov_ms else win_ms
        paths: List[str] = []
        i = 0
        start = 0
        length = len(audio)
        while start < length:
            seg = audio[start:start+win_ms]
            out = os.path.join(segments_dir, f"{base_name}_segment_{i:03d}.wav")
            seg.export(out, format="wav", parameters=["-ac","1","-ar",str(sr)])
            paths.append(out)
            i += 1
            start += step
        return paths

    # Fallback: pure librosa slicing
    y, file_sr = librosa.load(path_16k, sr=sr, mono=True)
    win_samples = win_s * sr
    ov_samples = ov_s * sr
    step = win_samples - ov_samples if win_samples > ov_samples else win_samples
    n = len(y)
    idx = 0
    start = 0
    out_paths: List[str] = []
    while start < n:
        end = min(start + win_samples, n)
        seg = y[start:end]
        out_path = os.path.join(segments_dir, f"{base_name}_segment_{idx:03d}.wav")
        sf.write(out_path, seg, sr)
        out_paths.append(out_path)
        idx += 1
        start += step
    return out_paths

def transcribe_segments_parallel(chunk_paths: List[str], stt_fn, language: str | None) -> List[dict]:
    """Transkripsi segmen secara paralel (ThreadPool) sampai STT_MAX_WORKERS."""
    if not chunk_paths:
        return []
    workers = max(1, settings.STT_MAX_WORKERS)
    results: List[dict] = [None] * len(chunk_paths)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(stt_fn, p, language): idx for idx, p in enumerate(chunk_paths)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {"file": chunk_paths[idx], "text": "", "error": str(e), "chunks": []}
    return results

def merge_overlap_text(segments: List[dict]) -> str:
    """Gabungkan teks segmen overlapped secara sederhana.

    Saat ini: sederhana join newline. (Bisa dikembangkan: deteksi duplikasi overlap.)
    """
    return "\n".join([s.get("text", "").strip() for s in segments if s and s.get("text")])
