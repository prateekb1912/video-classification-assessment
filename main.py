import os
import json
import faiss
import numpy as np

from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# some workarounds 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

model = WhisperModel("tiny",compute_type="float32")
openai_client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
EMBEDDING_MODEL = "text-embedding-3-small"

def get_yt_transcript(video_path, output_path):
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    with open(output_path, "w") as f:
        for segment in segments:
            f.write(json.dumps({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            }) + "\n")
    

def embed_and_save_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
        dimensions=1536
    )
    
    embeddings = np.array([embedding.embedding for embedding in response.data], dtype=np.float32)

    index = faiss.IndexFlatL2(1536)

    index.add(embeddings)
    faiss.write_index(index, "video_embeddings.index")
    
    return index

def search(index, query, k=5):
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL,
        dimensions=1536
    )
    
    query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

    similarities, indices = index.search(query_embedding, k)

    return similarities, indices

def main():
    video_path = "video.mp4"
    query = "what did laborers in ancient Egypt use?"

    if not os.path.exists("transcript.jsonl"):
        get_yt_transcript(video_path, "transcript.jsonl")
    
    with open("transcript.jsonl", "r") as f:
        segments = [json.loads(line) for line in f]

    if not os.path.exists("video_embeddings.index"):
        index = embed_and_save_chunks(segments)
    else:
        index = faiss.read_index("video_embeddings.index")

    _, indices = search(index, query)

    most_relevant_segment = segments[indices[0][0]]

    print(most_relevant_segment)

if __name__ == "__main__":
    main()