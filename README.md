
## 🚗 Real-Time Drowsiness Detection System Using Python & CNN

This project focuses on detecting **driver drowsiness in real-time** using a webcam. It combines **OpenCV for face & eye detection** and a **CNN model** to classify eye states (open/closed). An alarm triggers when the system detects prolonged eye closure.

---

### 📁 Project Structure

* **`haar cascade files/`** – Contains XML files for face and eye detection (Haar cascades).
* **`models/cnnCat2.h5`** – Pretrained CNN model to classify eye status.
* **`alarm.wav`** – Alarm sound triggered when drowsiness is detected.
* **`Model.py`** – CNN training script.
* **`Drowsiness detection.py`** – Main script to run the detection system.

---

### ⚙️ How It Works – Step by Step

#### 1️⃣ Capture Frames from Webcam

```python
cap = cv2.VideoCapture(0)
```

Each frame from the webcam is continuously captured for processing.

#### 2️⃣ Face & Eye Detection (ROI Creation)

Convert each frame to grayscale and use Haar cascades:

```python
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
```

Detect face and eyes using `detectMultiScale()`.

#### 3️⃣ Eye Extraction and Preprocessing

Extract the eye regions, resize to **24x24**, normalize, and reshape for CNN input:

```python
eye = cv2.resize(eye_gray, (24, 24)) / 255
eye = np.expand_dims(eye.reshape(24, 24, -1), axis=0)
```

#### 4️⃣ Predict Eye Status

Load the trained model and predict:

```python
model = load_model('cnnCat2.h5')
pred = model.predict_classes(eye)
```

* `1` → Open
* `0` → Closed

#### 5️⃣ Drowsiness Scoring & Alarm

* A **score** increments for each frame where both eyes are closed.
* When `score > 15`, trigger the **alarm** and flash a red alert rectangle.

```python
if score > 15:
    sound.play()
    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)
```

---

### 🧪 Running the System

From the terminal:

```bash
python "drowsiness detection.py"
```

Make sure the webcam is accessible. The system will start detecting and alerting in real time.

---

### 🖼️ Output Examples

* **Open Eyes Detected** → Displays “Open”
* **Closed Eyes Detected** → Displays “Closed”
* **Sleep Alert** → Triggers alarm with red overlay

---

### ✅ Summary

This project is a practical implementation of **computer vision + deep learning**:

* Uses **OpenCV** for real-time detection.
* Implements a **CNN classifier** for eye state prediction.
* Alerts the driver with an **audio alarm** when drowsiness is detected.

A compact, effective safety system for smart vehicles, research, or personal experimentation.
