# SignSpeak-Real-Time-Sign-Language-to-Speech-Converter
Real-Time Indian Sign Language Recognition and Translation System using Deep Learning
# SignSpeak: Real-Time Sign Language to Speech Converter

## Project Overview
SignSpeak is a real-time sign language translation system that converts hand gestures into text and speech. The system uses computer vision and deep learning techniques to recognize sign language gestures and translate them into meaningful spoken output.

This project aims to help bridge the communication gap between hearing-impaired individuals and others.

---

## Features
- Real-time hand gesture recognition
- Sign-to-text translation
- Text-to-speech output
- Webcam-based gesture detection
- Deep learning-based gesture classification
- User-friendly graphical interface

---

## Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Scikit-learn
- Tkinter GUI

---

## System Architecture

Sign Gesture Input (Webcam)
↓  
MediaPipe Hand Landmark Detection  
↓  
Feature Extraction  
↓  
Deep Learning Model (CNN + LSTM + Attention)  
↓  
Gesture Classification  
↓  
Text Output  
↓  
Text-to-Speech Conversion  

---

## Project Structure
SignSpeak-Real-Time-Sign-Language-to-Speech-Converter
│
├── Dataset
├── model
├── testVideo
│
├── GUI.py
├── train.py
├── test.py
├── test1.py
├── attention.py
├── requirements.txt
├── run.bat
│
└── README.md

---

## How to Run the Project

### 1. Clone the Repository
git clone https://github.com/Vyshnavee05d7/SignSpeak-Real-Time-Sign-Language-to-Speech-Converter.git


### 2. Navigate to the Project Folder


cd SignSpeak-Real-Time-Sign-Language-to-Speech-Converter


### 3. Install Dependencies


pip install -r requirements.txt


### 4. Run the Application


python GUI.py


---

## Dataset
The dataset contains gesture images representing different Indian Sign Language words.

Example labels used:
- I
- apple
- can
- get
- good
- help
- how
- like
- love
- yes
- no
- thank-you
- sorry
- want
- you
- your

---

## Results and Metrics
The system successfully recognizes sign gestures and converts them into text and speech output.

Evaluation metrics used:
- Accuracy
- Loss Function
- BLEU Score for translation quality

---

## Future Improvements
- Support for full sentence recognition
- Mobile application integration
- Larger gesture dataset
- Faster real-time detection

---

## Author
Major Project C-16 
B.Tech Computer Science Engineering  
Shri Vishnu Engineering College for Women

---

## License
This project is developed for academic purposes.
