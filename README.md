# 🚦 Real-Time Traffic Rule Violation Detection Using Deep Learning

An AI-powered smart surveillance system that detects traffic rule violations from CCTV / road traffic videos using **YOLOv8**, **OpenCV**, **Flask**, and **OCR**.

This project automatically identifies vehicles, monitors traffic signals, detects red-light crossing violations, captures evidence screenshots, extracts probable number plate regions, and displays all results through a clean web interface.

---

# 📌 Project Overview

Traffic rule violations are a major cause of road accidents and congestion. Manual monitoring through CCTV systems is time-consuming and inefficient.

This project provides an automated solution using Computer Vision and Deep Learning.

The system can:

* Detect vehicles in uploaded traffic videos
* Simulate traffic signals (Red / Green)
* Detect red-light crossing violations
* Capture proof screenshots
* Extract probable number plate regions
* Apply OCR for plate scanning
* Generate processed result video
* Show results in a modern browser UI

---

# 🎯 Objectives

* Automate traffic monitoring
* Reduce manual surveillance effort
* Improve rule enforcement
* Enable future e-challan systems
* Demonstrate Smart City AI applications

---

# 🧠 Technologies Used

## Frontend

* HTML5
* CSS3
* Flask Templates (Jinja2)

## Backend

* Python
* Flask

## AI / Computer Vision

* YOLOv8 (Ultralytics)
* OpenCV

## OCR

* EasyOCR

## Video Conversion

* MoviePy

---

# ⚙️ Features

## ✅ Vehicle Detection

Detects:

* Car
* Motorcycle
* Bus
* Truck

Using pretrained YOLOv8 model.

---

## 🚦 Traffic Signal Logic

The system simulates traffic light states:

* RED
* GREEN

Signal changes automatically after fixed frames.

---

## 🚫 Violation Detection

If a vehicle crosses the stop line during RED signal:

* Violation detected
* Count increases
* Alert shown on frame

---

## 📸 Evidence Capture

For every violation:

* Full screenshot saved
* Vehicle snapshot recorded

---

## 🔍 Number Plate Region Extraction

Instead of OCR on full image, system crops likely number plate region from violating vehicle.

This improves readability.

---

## 🔠 OCR Plate Scanning

EasyOCR attempts to read number plate text.

Output examples:

* DL8CAF5032
* UP16AB2045
* Not Detected

(Depends on camera quality)

---

## 🎥 Processed Output Video

System generates processed downloadable video containing:

* Bounding boxes
* Vehicle labels
* Traffic signal
* Stop line
* Violation counter
* Violation alerts

---

## 🌐 Web Dashboard

User-friendly interface with:

* Video upload
* Process button
* Result video
* Download option
* Violation cards
* Plate crop preview
* OCR results
* Statistics cards

---

# 🗂️ Project Structure

```text
traffic-ai/
│── app.py
│── requirements.txt
│── README.md
│
├── static/
│   ├── style.css
│   └── result.mp4
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── evidence/
│   ├── violation_1.jpg
│   ├── plate_1.jpg
│   └── ...
```

---

# 🛠️ Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/traffic-rule-violation-detection.git
cd traffic-rule-violation-detection
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

OR

```bash
pip install flask ultralytics opencv-python easyocr moviepy
```

## Step 3: Run Project

```bash
python app.py
```

## Step 4: Open Browser

```text
http://127.0.0.1:5000
```

---

# 📥 How To Use

1. Open website
2. Upload traffic CCTV video
3. Click **Process Video**
4. Wait for AI analysis
5. View:

   * Result video
   * Violation screenshots
   * Plate crops
   * OCR results

---

# 🧠 How It Works

## Step 1: Input Video

User uploads road traffic video.

## Step 2: Frame Processing

Video split into frames.

## Step 3: Vehicle Detection

YOLOv8 detects vehicles frame-by-frame.

## Step 4: Signal Simulation

Traffic light toggles between RED/GREEN.

## Step 5: Line Crossing Logic

If vehicle crosses line during RED:

Violation triggered.

## Step 6: Evidence Storage

System saves:

* Full screenshot
* Plate crop

## Step 7: OCR

EasyOCR scans plate region.

## Step 8: Output Generation

Processed video rendered and shown in browser.

---

# 📈 Example Output

```text
Violations Detected: 3

Vehicle 1:
Plate Scan: UP16AB2045

Vehicle 2:
Plate Scan: Not Detected
```

---

# 🎓 Academic Relevance

Useful for subjects:

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Smart City Systems
* Traffic Management
* Machine Learning Applications

---

# 🚀 Future Enhancements

* Real traffic signal integration
* Live CCTV camera support
* Helmet detection
* Seatbelt detection
* Triple riding detection
* Speed violation detection
* Automatic challan generation
* Cloud database storage
* Email/SMS alerts

---

# ⚠️ Limitations

* OCR depends on image clarity
* Far CCTV cameras reduce plate accuracy
* Current signal logic is simulated
* Uses uploaded videos (not live stream)

---

# 💡 Why This Project Is Valuable

This project demonstrates real-world AI deployment using:

* Deep Learning
* Real-time Detection
* Automation
* Law Enforcement Technology
* Smart Infrastructure

---

# 📚 Dependencies

```text
Flask
Ultralytics
OpenCV
EasyOCR
MoviePy
Python 3.8+
```

---

# 👨‍💻 Author

**Ujjwal Tayal**
BTech CSE (AI & ML)

---

# ⭐ If You Like This Project

Star the repository and support the work.

---

# 📜 License

This project is for educational and academic purposes.
