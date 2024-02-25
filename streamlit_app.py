import random
import base64
import requests
import numpy as np
import cv2
import webbrowser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import firebase_admin
from firebase_admin import credentials, auth
# 
# CLIENT_ID = '1973ff0394be4f7d8e695b37b4ebfa56'
# CLIENT_SECRET = '7a5887b8d98e4877a5fec231e436b457'

# if not firebase_admin._apps:
#     cred = credentials.Certificate(r"C:\Users\poojitha\Downloads\music-player-4dadd-firebase-adminsdk-x2i3x-93d0383e17.json")
#     firebase_admin.initialize_app(cred)

emotion_mapping = {
    "happy": [["Gaaju Bomma", "Hesham Abdul Wahab"], ["Na Roja Nuvve", "Hesham Abdul Wahab"]],
    "sad": [["Jabilli Kosam-Male", "S. P. Balasubrahmanyam"], ["Povodhe Prema", "Yuvan Shankar Raja"]],
    # Add more emotions and song lists as needed
}

def get_random_song(emotion):
    emotion = emotion.lower()
    if emotion in emotion_mapping:
        song_artist_pair = random.choice(emotion_mapping[emotion])
        return song_artist_pair
    else:
        return None

def get_track_id(emotion, access_token, emotion_mapping):
    emotion = emotion.lower()
    emotion = emotion.strip()
    if emotion in emotion_mapping:
        song_artist_pair = random.choice(emotion_mapping[emotion])
        song_name, artist_name = song_artist_pair
        base_url = "https://api.spotify.com/v1/search"
        headers = {"Authorization": f"Bearer {access_token}"}

        params = {
            "q": f"{song_name} {artist_name}",
            "type": "track",
            "limit": 1
        }

        response = requests.get(base_url, params=params, headers=headers)
        data = response.json()

        if "tracks" in data and "items" in data["tracks"] and data["tracks"]["items"]:
            track_id = data["tracks"]["items"][0]["id"]
            return track_id
        else:
            return None
    else:
        return None

def play_song_by_emotion(emotion, access_token, emotion_mapping):
    emotion = emotion.lower()
    emotion = emotion.strip()
    song_artist_pair = get_random_song(emotion)

    if song_artist_pair:
        song_name, artist_name = song_artist_pair
        track_id = get_track_id(emotion, access_token, emotion_mapping)

        if track_id:
            play_url = f"https://open.spotify.com/track/{track_id}?autoplay=true"
            webbrowser.open(play_url)
        else:
            st.write(f"Track ID not found for {song_name} by {artist_name}")
    else:
        st.write(f"No songs mapped for emotion: {emotion}")

def add_song_to_library(artist_name, song_name, mood, emotion_mapping):
    mood = mood.lower().strip()

    if mood not in emotion_mapping:
        emotion_mapping[mood] = []

    emotion_mapping[mood].append([song_name, artist_name])

def add_song_form(emotion_mapping):
    st.header("Add Song to Library")
    st.subheader("Provide information about the song")

    artist_name = st.text_input("Artist Name:")
    song_name = st.text_input("Song Name:")
    mood = st.text_input("Mood:")

    add_song_button = st.button("Add Song to Library")

    if add_song_button:
        add_song_to_library(artist_name, song_name, mood, emotion_mapping)
        st.success(f"Added {song_name} by {artist_name} to the {mood} playlist.")
def add_song_to_library(artist_name, song_name, mood, emotion_mapping):
    mood = mood.lower().strip()

    if mood not in emotion_mapping:
        emotion_mapping[mood] = []

    emotion_mapping[mood].append({"song_name": song_name, "artist_name": artist_name})

def add_song_form(emotion_mapping):
    st.header("Add Song to Library")
    st.subheader("Provide information about the song")

    artist_name = st.text_input("Artist Name:")
    song_name = st.text_input("Song Name:")
    mood = st.text_input("Mood:")

    add_song_button = st.button("Add Song to Library")

    if add_song_button:
        add_song_to_library(artist_name, song_name, mood, emotion_mapping)
        st.success(f"Added {song_name} by {artist_name} to the {mood.capitalize()} playlist.")

    # Display the updated emotion_mapping
    st.write("Updated Library:")
    st.write(emotion_mapping)



def get_access_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(f'{CLIENT_ID}:{CLIENT_SECRET}'.encode('utf-8')).decode('utf-8')}
    data = {'grant_type': 'client_credentials'}

    response = requests.post(auth_url, headers=headers, data=data)
    if response.status_code == 200:
        access_token = response.json().get('access_token')
        return access_token
    else:
        raise Exception(f"Failed to obtain access token: {response.text}")

# Load your pre-trained emotion detection model and cascade classifier
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.song_played = False
        self.start_time = time.time()
        self.no_face_warning_shown = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)

            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0 and not self.song_played:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)

                access_token = get_access_token()
                play_song_by_emotion(output, access_token, emotion_mapping)

                self.song_played = True

            label_position = (x, y - 10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

    def show_no_face_warning(self):
        if not self.no_face_warning_shown and time.time() - self.start_time > 10:
            st.warning("No face detected for 6 seconds. Please make sure your face is visible.")
            self.no_face_warning_shown = True

def main():
    st.title("Emotion-based music player")
    activities = ["Home", "Live Face Emotion Detection", "About", "Create Your Library"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by poojitha and team
            [LinkedIn](https://www.linkedin.com/in//)""")

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#FC4C02;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                            Start Your Real Time Face Emotion Detection.
                             </h4>
                             </div>
                             </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)
        # st.write("""
        # * An average human spends about 10 to 15hrs a day staring at a computer screen, during which our facial expressions keep on changing. 
        # * Sometimes we laugh, sometimes we cry, sometimes we get angry, and sometimes get scared by our face when the camera turns on accidentally.
        # * But ever wondered; whether the computer that we give all this attention to is even capable of recognizing these emotions?
        # 
        # Let's find out...
        # 1. Click the dropdown list in the top left corner and select Live Face Emotion Detection.
        # 2. This takes you to a page which will tell if it recognizes your emotions.
        #          """)

    elif choice == "Live Face Emotion Detection":

        st.header("Webcam Live Feed")

        st.subheader('''


                Welcome to the other side of the SCREEN!!!


                * Get ready with all the emotions you can express. 


                ''')

        # st.write("1. Click Start to open your camera and give permission for prediction")
        #
        # st.write("2. This will predict your emotion.")
        #
        # st.write("3. When you're done, click stop to end.")

        video_transformer = VideoTransformer()

        webrtc_streamer(key="example", video_processor_factory=lambda: video_transformer)

        time.sleep(5)

        video_transformer.song_played = False

        while True:
            video_transformer.show_no_face_warning()
    elif choice == "Create Your Library":
        add_song_form(emotion_mapping)

if __name__ == "__main__":
    main()
