from flask import Blueprint, request, jsonify

import whisper
import ffmpeg
import ollama
import subprocess
import os
import cv2
import torch
import ffmpeg
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from paddleocr import PaddleOCR
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from utilities.keyframe_sel import *

summarize_routes = Blueprint('summarize_routes', __name__)

@summarize_routes.route('/extract/<filename>', methods=['POST', 'GET'])
def extract_imgs(filename):
    VIDEO_PATH = 'uploads/' + filename
    OUTPUT_DIR = "uploads"

    print("Detecting scenes...")
    dynamic_threshold = estimate_dynamic_threshold(VIDEO_PATH)
    video_manager = VideoManager([VIDEO_PATH])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=dynamic_threshold*2.5))
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scenes = scene_manager.get_scene_list(video_manager.get_base_timecode())
    video_manager.release()
    scenes = merge_short_scenes(scenes)
    print(f"Number of scenes after merging: {len(scenes)}")

    # ---------------- VIDEO CAPTURE ----------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    all_keyframes = []
    all_embeddings = []

    # ---------------- PROCESS SCENES ----------------
    for i, (start, end) in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        start_sec = max(0, start.get_seconds() - 0.5)
        end_sec = end.get_seconds() + 0.5
        duration = end_sec - start_sec
        if duration < 1.0:
            continue

        # --- Extract scene audio and transcribe ---
        audio_path = f"scene_{i}.wav"
        (
            ffmpeg.input(VIDEO_PATH, ss=start_sec, t=duration)
                .output(audio_path, acodec="pcm_s16le", ar=16000, ac=1)
                .run(overwrite_output=True, quiet=True)
        )
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"].strip()
        os.remove(audio_path)
        if not transcript:
            continue
        summary = summarize_with_llama(transcript)
        text_chunks = [summary]

        # --- Sample frames ---
        start_frame = int(start.get_frames())
        end_frame = int(end.get_frames())
        step = max(1, (end_frame - start_frame) // (MAX_FRAMES_PER_SCENE*2))
        scene_frames, scene_embeddings = [], []

        for f in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret or not has_enough_text(frame):
                continue

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                img_emb = clip_model.get_image_features(**image_inputs)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

            # Compute max similarity against transcript chunks
            max_score = -1
            for chunk in text_chunks:
                tokens = clip_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_CLIP_TOKENS).to(DEVICE)
                with torch.no_grad():
                    text_emb = clip_model.get_text_features(**tokens)
                    text_emb /= text_emb.norm(dim=-1, keepdim=True)
                score = cosine_similarity(img_emb.cpu().numpy(), text_emb.cpu().numpy())[0][0]
                max_score = max(max_score, score)

            if max_score >= CLIP_SCORE_THRESHOLD:
                scene_frames.append((frame, max_score))  # store frame + score
                scene_embeddings.append(img_emb)

        # --- Select diverse frames ---
        selected_frames = select_diverse_frames(scene_frames, scene_embeddings)
        all_keyframes.extend(selected_frames)
        all_embeddings.extend(scene_embeddings[:len(selected_frames)])

    cap.release()

    # ---------------- SAVE KEYFRAMES ----------------
    print(f"Saving {len(all_keyframes)} keyframes to {OUTPUT_DIR}...")
    for idx, (frame, score) in enumerate(all_keyframes):
        out_path = os.path.join(OUTPUT_DIR, f"keyframe_{idx}_score{score:.2f}.jpg")
        cv2.imwrite(out_path, frame)

    print("Done!")

    return jsonify({
        'status': 'Success',
        'frames_saved': len(all_keyframes)
    })

@summarize_routes.route('/summarize/<filename>', methods=['POST', 'GET'])
def note(filename):
    try:
        model = whisper.load_model('base')
        processed_audio = 'processed_audio.wav'

        uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        input_file = os.path.abspath(os.path.join(uploads_dir, filename))

        print("Filename: " + filename)

        (
            ffmpeg.input(input_file)
            .output(processed_audio, acodec="pcm_s16le", ar=16000, ac=1)
            .run(overwrite_output=True)
        )

        result = model.transcribe(processed_audio)

        with open("uploads/transcript.md", "w", encoding='utf-8') as f:
            for segment in result['segments']:
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()

                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
                f.write(f"[{start_time}] {text}\n")

        os.remove(processed_audio)

        llama_note = ollama.chat(
            model='llama3.1',
            messages=[{
                'role': 'user',
                'content': f"You are an expert note-taker and knowledge organizer. Analyze the following transcript and generate a detailed tree-structured mindmap in valid JSON. The mindmap must have a single root node with a descriptive title summarizing the entire transcript (prefixed with 🌳). From the root, branch into main topics (📌), subtopics (🔹), details (✨), and examples (📝) or explanations. Preserve all crucial and relevant information without oversimplifying. Expand all details (`✨`) into clear, student-style notes**: complete sentences, explanations, context, and reasoning. The result should feel like comprehensive lecture notes rather than short keywords. Each node must include 'title' (with emoji) and 'children' (array of child nodes, or empty if none). Output only valid JSON. Example format: {{'title': '🌳 [Descriptive Title]','children':[{{'title':'📌 Main Topic','children':[{{'title':'🔹 Subtopic','children':[{{'title':'✨ Detail','children':[]}}]}}]}}]}} Every node must include 'title' (not 'text') and 'children' (array, empty if none). Transcript: \"\"\"{result['text']}\"\"\""
            }]
        )

        llama_summary = ollama.chat(
            model='llama3.1',
            messages=[{
                'role': 'user',
                'content': f"You are given this transcript {result['text']}. Make a detailed summary with supporting details while not losing any crucial information. Avoid oversimplifying or merging distinct ideas into vague statements. Also preserve technical terms, names, numbers, dates and examples."
            }]
        )

        output_summary = llama_summary['message']['content']
        output_note = llama_note['message']['content']

        start = output_note.find("{")
        end = output_note.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output.")
        json_str = output_note[start:end]

        with open("uploads/mindmap.txt", "w", encoding='utf-8') as f:
            f.write(output_note)

        with open("uploads/summary.txt", "w", encoding='utf-8') as f:
            f.write(output_summary)

        return {
            "notes": json_str,
            "summary": output_summary,
        }
    except Exception as e:
        return (f'Error {e}')