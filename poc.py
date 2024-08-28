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