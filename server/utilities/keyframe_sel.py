import ollama
import cv2
import numpy as np
from paddleocr import PaddleOCR
import torch
from sklearn.metrics.pairwise import cosine_similarity
import whisper
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CLIP_TOKENS = 77
MIN_SCENE_DURATION = 2.0
DUPLICATE_THRESH = 0.8
MAX_FRAMES_PER_SCENE = 3
OCR_MIN_WORDS = 10
FRAMES_PER_SCENE = 5
CLIP_SCORE_THRESHOLD = 0.25  # Relevance threshold for frame selection

ocr = PaddleOCR(lang='en')

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def summarize_with_llama(transcript, max_new_tokens=64):
    summary = ollama.chat(
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': f"Summarize the following transcript into one short complete sentence. Focus on describing slides, bullet points, diagrams, or other visual content likely shown: {transcript}"
        }]
    )
    return summary['message']['content']

# ---------------- UTILITIES ----------------
def estimate_dynamic_threshold(video_path, num_frames=300, percentile=80):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 1:
        cap.release()
        return 15.0

    # Pick evenly spaced frame indices across the video
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    diffs = []
    prev_gray = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            diffs.append(diff)

        prev_gray = gray

    cap.release()

    if not diffs:  # fallback if no frames were processed
        return 15.0

    threshold = np.percentile(diffs, percentile)
    return np.clip(threshold, 5.0, 50.0)

def merge_short_scenes(scenes, min_duration=MIN_SCENE_DURATION):
    if not scenes:
        return []
    merged = []
    prev_start, prev_end = scenes[0]
    for start, end in scenes[1:]:
        prev_duration = prev_end.get_seconds() - prev_start.get_seconds()
        if prev_duration < min_duration:
            prev_end = end
        else:
            merged.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    merged.append((prev_start, prev_end))
    return merged

def has_enough_text(frame, min_words=OCR_MIN_WORDS):
    result = ocr.ocr(frame)
    text_content = " ".join([line[1][0] for r in result for line in r])
    return len(text_content.strip().split()) >= min_words

def select_diverse_frames(
    frames_with_scores, embeddings,
    max_frames=MAX_FRAMES_PER_SCENE,
    duplicate_thresh=DUPLICATE_THRESH
):
    # --- 1. Sort frames by score (highest first) ---
    scored_embs = list(zip(frames_with_scores, embeddings))
    scored_embs.sort(key=lambda x: x[0][1], reverse=True)

    selected = []
    selected_embs = []

    # --- 2. Iterate in order of importance ---
    for (frame, score), emb in scored_embs:
        keep = True
        for e in selected_embs:
            sim = cosine_similarity(emb.cpu().numpy(), e.cpu().numpy())[0][0]
            if sim > duplicate_thresh:  # too similar to an already kept frame
                keep = False
                break

        if keep:
            selected.append((frame, score))
            selected_embs.append(emb)

        if len(selected) >= max_frames:
            break

    return selected
