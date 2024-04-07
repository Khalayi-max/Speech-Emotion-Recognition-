# Speech Emotion Recognition

Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion.
SER is tough because emotions are subjective and annotating audio is challenging.
In this Python project, we recognize emotions from speech. The model delivers an accuracy of 73.96%. That’s good enough for us !!.


* **Model:** Build using an MLPClassifier(Multi-layer Perceptron classifier). This model optimizes the log-loss function using LBFGS or stochastic gradient descent.Unlike other classification algorithms such as Support Vectors or Naive Bayes Classifier, MLPClassifier relies on an underlying Neural Network to perform the task of classification.
* **Dataset:** RAVDESS dataset(Ryerson Audio-Visual Database of Emotional Speech and Song dataset).This dataset has 7356 files rated by 247 individuals 10 times on emotional validity, intensity, and genuineness. 
* **Prerequisities:** 
  * Librosa   : It is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.
  * Soundfile : SoundFile can read and write sound files. Data can be written to the file using soundfile.write(), or read from the file using soundfile.read(). SoundFile can open all file formats that libsndfile supports, for example WAV, FLAC, OGG and MAT files.
  * Numpy     : NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
  * Sklearn   : We will use Python's Scikit-Learn library to create our neural network that performs this classification task; importing the MLPClassifier class from the sklearn.neural_network library
  * Pyaudio   : PyAudio provides Python bindings for PortAudio, the cross-platform audio I/O library. With PyAudio, you can easily use Python to play and record audio on a variety of platforms.
  * AudioSegment : Returns a list of AudioSegments, each of which is all the sound during this AudioSegment’s duration from a particular source. That is, if there are several overlapping sounds in this AudioSegment, this method will return one AudioSegment object for each of those sounds.
  * Streamlit: Streamlit is used for hosting and app sharing.

Live link of the project goes [here](https://share.streamlit.io/atinder01/speechemotionrecognition/main/main.py)

For more insights, you can refer to this video:

https://user-images.githubusercontent.com/67895402/135711745-8afcddb5-1613-4a8a-911f-2b54409e7001.mp4

### FlowChart:

![FlowChart](https://user-images.githubusercontent.com/67895402/134717714-7db8bc99-f36d-47e9-a0db-d1f59dcac8e0.png)

### Screenshots:

![image](https://user-images.githubusercontent.com/67895402/134714921-86f32d02-7fd6-4953-8e48-a9a1e9d52af8.png)
![image](https://user-images.githubusercontent.com/67895402/134713909-d90c8479-bf0b-41c4-9ef8-db371f8aef6a.png)
![image](https://user-images.githubusercontent.com/67895402/135554722-9a83f7c0-27de-40a4-b175-aa3a0da44270.png)
