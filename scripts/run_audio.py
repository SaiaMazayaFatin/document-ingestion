import os, sys, uuid

# Ensure project root (parent of this scripts directory) is on sys.path when executed directly
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.schemas import PipeState
from pipelines.graph_audio import build_graph

AUDIO_DIR = "data/raw/audio"
VALID_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

def is_raw_audio_file(filename: str) -> bool:
    """Filter only original audio files, skip intermediate/processed files."""
    name = filename.lower()
    # Must have valid extension
    if not name.endswith(VALID_EXTENSIONS):
        return False
    # Skip resampled files
    if ".16k.wav" in name:
        return False
    # Skip segment files (pattern: .000.wav, .001.wav etc)
    if any(f".{i:03d}.wav" in name for i in range(1000)):
        return False
    return True

def new_state(path: str):
    lang = "en" if "_en" in path.lower() else ("id" if "_id" in path.lower() else "auto")
    return PipeState(
        modality="audio",
        doc_id=f"doc_{uuid.uuid4().hex[:8]}",
        title=os.path.splitext(os.path.basename(path))[0],
        language=lang,
        file_path=path,
        file_name=os.path.basename(path),
        transcript_raw_segments=[], transcript_full="", transcript_clean="",
        chunks=[], extraction={}
    )

if __name__ == "__main__":
    graph = build_graph()
    all_files = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))]
    raw_files = [f for f in all_files if is_raw_audio_file(f)]
    
    if not raw_files:
        print(f"Tidak ada file audio mentah di {AUDIO_DIR}")
        print(f"File ditemukan: {all_files}")
        raise SystemExit(1)
    
    for filename in raw_files:
        full_path = os.path.join(AUDIO_DIR, filename)
        print(f"[AUDIO] Processing: {filename}")
        out = graph.invoke(new_state(full_path))
        print("✅ Done:", out["doc_id"])
