import os, uuid
from models.schemas import PipeState
from pipelines.graph_audio import build_graph

AUDIO_DIR = "data/raw/audio"

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
    files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.lower().endswith((".wav",".mp3",".m4a"))]
    if not files:
        print(f"Tidak ada file audio di {AUDIO_DIR}"); raise SystemExit(1)
    for f in files:
        print(f"[AUDIO] {f}")
        out = graph.invoke(new_state(f))
        print("✅ Done:", out["doc_id"])
