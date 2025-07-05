import os
# Fix OpenMP library conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

model = WhisperModel("small")

def get_yt_transcript(video_path):
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    return segments

def save_transcript(segments, output_path):
    with open(output_path, "w") as f:
        for segment in segments:
            f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n")

def main():
    video_path = "video.mp4"
    transcript_segments = get_yt_transcript(video_path)
    save_transcript(transcript_segments, "transcript.txt")
    print("Transcript saved to transcript.txt")

if __name__ == "__main__":
    main()