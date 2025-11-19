# OncoVision AI Pro

Advanced Multi-Organ Tumor Detection & Clinical Analysis Platform

OncoVision AI Pro is a diagnostic-assistance system built for research and educational use.
It performs automated tumor detection on multiple organ systems using YOLO-based deep learning models, evaluates image quality, generates clinical-style reports, saves patient records, and provides analytics through a modern Gradio interface.

This project is designed as an end-to-end prototype for medical imaging AI workflows.

---

## Features

### 1. Multi-Organ Tumor Detection

Supports:

* Brain (MRI)
* Lung (CT)
* Liver (CT)
* Kidney (CT)

Each organ uses a dedicated model (custom YOLO weights). If custom weights are missing, the system safely falls back to a general YOLOv8 model.

### 2. Ensemble Prediction Pipeline

The platform can optionally run predictions at multiple confidence levels and merge them.
This improves detection sensitivity and robustness in challenging scans.

### 3. Clinical-Style Reporting

Every analysis generates:

* Primary diagnosis and severity stage
* Tumor size estimation in millimeters
* Risk stratification (Low / Moderate / High / Critical)
* Cost estimation (based on typical treatment protocols)
* Recommended treatment pathway
* Confidence metrics and thresholds

### 4. Image Quality Assessment

Automatic detection of:

* Motion blur
* Underexposure / overexposure
* Low contrast

This provides quick feedback on whether a scan is suitable for AI-based analysis.

### 5. Patient Records Management

The platform maintains:

* A complete scan history
* Timestamped doctor notes
* Filterable record views (by organ, risk level, or date range)

Records are saved to local CSV + JSON for immediate portability.

### 6. Batch Processing

Upload multiple CT/MRI images and run automated diagnostics in bulk.
A summary and complete table of results are generated instantly.

### 7. Analytics Dashboard

Displays:

* Organ-wise scan counts
* Risk distribution
* Recent activity
* Dataset summary statistics

Useful for operational tracking and research analysis.

### 8. Professional UI (No Emojis)

The interface uses:

* Gradio 4.x
* Clean typography (Inter)
* Soft clinical color palette
* Optional `simpleicons` SVG icons for branding consistency

---

## Installation

### Requirements

Python 3.9+

### Install dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install gradio ultralytics opencv-python numpy pandas simpleicons
```

### Add your models

Place model files in the project directory:

```
brain_yolo.pt
lung_yolo.pt
liver_yolo.pt
kidney_yolo.pt
```

If any of these are missing, the system will automatically fall back to a YOLOv8-Nano model.

---

## Running the Application

```
python app.py
```

This launches a Gradio web interface at:

```
http://127.0.0.1:7860
```

---

## Project Structure

```
.
├── app.py                     # Main application
├── brain_yolo.pt              # Custom YOLO model (optional)
├── lung_yolo.pt
├── liver_yolo.pt
├── kidney_yolo.pt
├── patient_records.csv        # Auto-generated
├── doctor_notes.json          # Auto-generated
├── scan_comparisons.json      # Reserved for future features
└── README.md
```

---

## Disclaimers

This project is strictly intended for:

* Research
* Prototyping
* Educational demonstrations
* Non-clinical experimentation

It is **not** a medically approved diagnostic tool.
All outputs must be reviewed by certified medical professionals before use in any real-world scenario.

---

## Future Enhancements (Roadmap)

* DICOM support with metadata extraction
* 3D CT/MRI volumetric processing
* UNet-based segmentation models
* PACS integration
* Role-based authentication for hospital settings
* Real-time comparative scan analysis

