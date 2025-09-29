
# 🚦 AI Traffic Violation Detection System

A computer vision–powered traffic monitoring system that detects and logs traffic violations in real-time using **YOLO**, **DeepSORT**, and a **Streamlit dashboard**.  

This project was built during a hackathon to showcase the potential of AI in **smart city solutions**.

---

## ✨ Features
- 🔍 **Object Detection**: Detects vehicles, pedestrians, and traffic lights using YOLOv8.  
- 🛑 **Violation Rules**:  
  - Red light violation  
  - Wrong lane detection  
  - Overspeeding (optional with calibration)  
- 📍 **Auto Camera Calibration** for speed & distance estimation.  
- 🎥 **Multi-object Tracking** with DeepSORT.  
- 💾 **Database Integration** with MongoDB (violations logged with timestamp, vehicle type, snapshot).  
- 📊 **Streamlit Dashboard** for live monitoring & violation reports.  

---

## 🖥️ Project Flow
1. 🎦 Upload or stream live video feed.  
2. 🧠 YOLO detects vehicles, people, and traffic lights.  
3. 📌 DeepSORT assigns unique IDs and tracks movement.  
4. ⚖️ Rule Engine checks for violations (red light, lane, speed).  
5. 📷 Snapshot + violation details stored in MongoDB.  
6. 📊 Dashboard displays real-time analytics.  

---

## 🚀 Tech Stack
- **Frontend**: Streamlit  
- **Backend**: Python (FastAPI optional)  
- **ML Model**: YOLOv8 (Ultralytics)  
- **Tracking**: DeepSORT  
- **Database**: MongoDB  
- **Others**: OpenCV, Numpy, Pandas  

---

## 📂 Project Structure
```bash
├── cv-module/
│   ├── detector.py             # YOLO + DeepSORT detection
│   ├── rule_engine.py          # Rules for traffic violations
│   ├── db.py                   # MongoDB helper functions
│   ├── dashboard_streamlit.py  # Streamlit dashboard
│   ├── utils.py                # Helper utilities
│   └── tools/
│       └── auto_calibrate.py   # Auto camera calibration
├── models/                     # Pretrained YOLO weights
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```   
## 🚀 Installation
```bash
# Clone the repo
git clone https://github.com/nikitajitendrasolanki/31_Codexcrew.git
cd 31_Codexcrew/cv-module

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
``` 
## ▶️ Usage
```bash
# Run Streamlit dashboard
streamlit run dashboard_streamlit.py
``` 
## 📸 Demo
---
![LANDING PAGE](images/UI.png)
![MODEL UI](images/MODEL.png)
![DETECTION EXAMPLE](images/DETECTION.png)
![DASHBOARD EXAMPLE](images/DASHBOARD.png)
![ANALYSIS EXAMPLE](images/ANALYSIS.png)
![AUDIT REPORT](images/AUDIT.png)
---

## 📜 License
MIT License – Free to use and modify.

