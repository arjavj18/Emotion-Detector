import streamlit as st
import os
import tempfile
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import torch
from scipy.special import softmax
from facenet_pytorch import MTCNN
from transformers import (AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig, 
                          AutoModelForAudioClassification, pipeline)
from PIL import Image
import librosa
import whisper
import pandas as pd
import matplotlib.pyplot as plt

# Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load models and setup
@st.cache_resource
def load_models():
    mtcnn = MTCNN(margin=10, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=False, device=device)
    extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
    model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
    modelDom = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Dominance", trust_remote_code=True)
    modelVal = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Valence", trust_remote_code=True)
    modelAro = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Arousal", trust_remote_code=True)
    whisper_model = whisper.load_model("base")
    text_classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    return mtcnn, extractor, model, modelDom, modelVal, modelAro, whisper_model, text_classifier

mtcnn, extractor, model, modelDom, modelVal, modelAro, whisper_model, text_classifier = load_models()

# Function definitions (detect_emotions, video_prob, Audio_emotion) go here
# ... (copy the functions from the original code)
def detect_emotions(image):
    """
    Detect emotions from a given image.
    Returns a tuple of the cropped face image and a
    dictionary of class probabilities.
    """
    temporary = image.copy()

    # Detect faces in the image using the MTCNN group model
    sample = mtcnn.detect(temporary)
    if sample[0] is not None:
        box = sample[0][0]

        # Crop the face
        face = temporary.crop(box)

        # Pre-process the face
        inputs = extractor(images=face, return_tensors="pt")

        # Run the image through the model
        outputs = model(**inputs)

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits,
                                                    dim=-1)

        # Retrieve the id2label attribute from the configuration
        config = AutoConfig.from_pretrained(
            "trpakov/vit-face-expression"
        )
        id2label = config.id2label

        # Convert probabilities tensor to a Python list
        probabilities = probabilities.detach().numpy().tolist()[0]

        # Map class labels to their probabilities
        class_probabilities = {
            id2label[i]: prob for i, prob in enumerate(probabilities)
        }

        return face, class_probabilities
    return None, None

# Choose a frame
def video_prob(video_data) :    
    skips = 2
    reduced_video = []

    for i in range(0, len(video_data), skips):
        reduced_video.append(video_data[i])

# Define a list of emotions
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# Create a list to hold the class probabilities for all frames
    all_class_probabilities = []

# Loop over video frames
    for i, frame in enumerate(reduced_video):
    # Convert frame to uint8
        frame = frame.astype(np.uint8)

    # Call detect_emotions to get face and class probabilities
        face, class_probabilities = detect_emotions(Image.fromarray(frame))
    
    # If a face was found
        if face is None:
            class_probabilities = {emotion: None for emotion in emotions}
        
    # Append class probabilities to the list
        all_class_probabilities.append(list(class_probabilities.values()))
    all_class_probabilities = np.asarray(all_class_probabilities)
    vid_prob = np.mean(all_class_probabilities,axis = 0)

    return softmax(vid_prob)

def Audio_emotion(model, audio_path):
    mean = model.config.mean
    std = model.config.std
    
    raw_wav, sr = librosa.load(audio_path, sr=model.config.sampling_rate)
    
    # normalize the audio by mean/std
    norm_wav = (raw_wav - mean) / (std + 0.000001)
    
    # generate the mask
    mask = torch.ones(1, len(norm_wav))
    
    # batch it (add dim)
    wavs = torch.tensor(norm_wav).unsqueeze(0)
    
    output = model(wavs, mask)
    return output.item() 

def split_video(input_path, output_folder, segment_duration):
    os.makedirs(output_folder, exist_ok=True)
    video = VideoFileClip(input_path)
    video_duration = int(video.duration)
    segments = []

    for index, start_time in enumerate(range(0, video_duration, segment_duration)):
        end_time = min(start_time + segment_duration, video_duration)
        segment_filename = os.path.join(output_folder, f"segment_{index:04d}.mp4")
        ffmpeg_extract_subclip(input_path, start_time, end_time, targetname=segment_filename)
        segments.append((segment_filename, start_time, end_time))

    return segments

def process_video_segment(segment_path):
    clip = VideoFileClip(segment_path)
    
    # Video processing
    video = clip.without_audio()
    video_data = np.array(list(video.iter_frames()))
    vid_prob = video_prob(video_data)

    # Audio processing
    audio_path = segment_path.replace('.mp4', '.mp3')
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    with torch.no_grad():
        pred_dom = Audio_emotion(modelDom, audio_path)
        pred_val = Audio_emotion(modelVal, audio_path)
        pred_aro = Audio_emotion(modelAro, audio_path)
    Audio_pred = [pred_dom, pred_val, pred_aro]

    # Text processing
    result = whisper_model.transcribe(audio_path)
    model_outputs = text_classifier(result["text"])
    text_emotions = [d["label"] for d in model_outputs[0]]
    text_preds = [d["score"] for d in model_outputs[0]]

    # Clean up temporary audio file
    os.unlink(audio_path)

    return vid_prob, Audio_pred, text_emotions, text_preds, result["text"]

def main():
    st.title("Video Emotion Analysis")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        segment_duration = st.slider("Segment Duration (seconds)", min_value=5, max_value=60, value=15, step=5)

        if st.button("Process Video"):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file temporarily
                temp_video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Split video into segments
                segments = split_video(temp_video_path, temp_dir, segment_duration)

                # Process each segment
                all_results = []
                progress_bar = st.progress(0)
                for i, (segment, start_time, end_time) in enumerate(segments):
                    vid_prob, Audio_pred, text_emotions, text_preds, transcription = process_video_segment(segment)
                    all_results.append((vid_prob, Audio_pred, text_emotions, text_preds, transcription, start_time, end_time))
                    progress_bar.progress((i + 1) / len(segments))

            # Create DataFrames for each segment
            for i, (vid_prob, Audio_pred, text_emotions, text_preds, transcription, start_time, end_time) in enumerate(all_results):
                st.subheader(f"Segment {i+1} ({start_time}s - {end_time}s)")
                
                df_vid = pd.DataFrame([vid_prob], columns=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
                df_audio = pd.DataFrame([Audio_pred], columns=["Dominance", "Valence", "Arousal"])
                df_text = pd.DataFrame([text_preds], columns=text_emotions)

                df_vid = df_vid.add_prefix('vid_')
                df_audio = df_audio.add_prefix('audio_')
                df_text = df_text.add_prefix('text_')

                df = pd.concat([df_text, df_audio, df_vid], axis=1)

                st.dataframe(df)

                st.text("Transcription:")
                st.write(transcription)

                # Visualization
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

                df_vid.T.plot(kind='bar', ax=ax1, title='Video Emotions')
                df_audio.T.plot(kind='bar', ax=ax2, title='Audio Emotions')
                df_text.T.plot(kind='bar', ax=ax3, title='Text Emotions')

                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()