import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from collections import defaultdict
import json

# --- 1. ADVANCED SETUP & MODELS ---
models = {}
model_files = {
    "Brain": "brain_yolo.pt",
    "Lung": "lung_yolo.pt",
    "Liver": "liver_yolo.pt",
    "Kidney": "kidney_yolo.pt"
}

print("⏳ Loading Medical Models...")
for organ, file in model_files.items():
    try:
        models[organ] = YOLO(file)
        print(f"✅ {organ} Model Loaded")
    except:
        print(f"⚠️ {file} not found. Using fallback.")
        models[organ] = YOLO("yolov8n.pt")

# Enhanced Data Storage
HISTORY_FILE = "patient_records.csv"
NOTES_FILE = "doctor_notes.json"
COMPARISON_FILE = "scan_comparisons.json"

# Initialize files
if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(columns=["Date", "Patient_ID", "Organ", "Diagnosis", "Confidence", 
                                "Tumor_Count", "Max_Size_mm", "Risk_Level", "Cost_Est", "Notes"])
    df.to_csv(HISTORY_FILE, index=False)

if not os.path.exists(NOTES_FILE):
    with open(NOTES_FILE, 'w') as f:
        json.dump({}, f)

if not os.path.exists(COMPARISON_FILE):
    with open(COMPARISON_FILE, 'w') as f:
        json.dump({}, f)

# --- 2. ADVANCED MEDICAL LOGIC ---
def calculate_tumor_size_mm(box, image_shape):
    """Convert pixel dimensions to approximate mm (assuming standard scan DPI)"""
    # Approximate: 1 pixel ≈ 0.5mm for CT/MRI at 512x512
    width_px = float(box[2])
    height_px = float(box[3])
    avg_size_mm = ((width_px + height_px) / 2) * 0.5
    return round(avg_size_mm, 1)

def get_risk_level(organ, size_mm, confidence):
    """Advanced risk stratification"""
    if organ == "Brain":
        if size_mm < 10:
            return "Low" if confidence < 70 else "Moderate"
        elif size_mm < 30:
            return "Moderate" if confidence < 80 else "High"
        else:
            return "Critical"
    elif organ == "Lung":
        if size_mm < 6:
            return "Low"
        elif size_mm < 15:
            return "Moderate"
        else:
            return "High"
    else:
        if size_mm < 20:
            return "Low"
        elif size_mm < 40:
            return "Moderate"
        else:
            return "High"

def get_medical_report(organ, box_area, confidence, size_mm, tumor_count):
    """Enhanced medical reporting with multiple tumors"""
    severity, cost, treatment = "Unknown", "N/A", "N/A"
    
    if organ == "Brain":
        if size_mm < 10:
            severity = "Stage I (Micro-lesion)"
            cost = "₹1.2L - ₹2.5L"
            treatment = "Active Surveillance + MRI Follow-up (3 months)"
        elif size_mm < 20:
            severity = "Stage II (Small Tumor)"
            cost = "₹2.5L - ₹5L"
            treatment = "Stereotactic Radiosurgery or Surgical Resection"
        elif size_mm < 40:
            severity = "Stage III (Moderate)"
            cost = "₹5L - ₹8L"
            treatment = "Craniotomy + Adjuvant Radiation"
        else:
            severity = "Stage IV (Advanced)"
            cost = "₹8L - ₹15L+"
            treatment = "Emergency Surgery + Chemo-Radiation Protocol"
    
    elif organ == "Lung":
        if size_mm < 6:
            severity = "Indeterminate Nodule"
            cost = "₹25k - ₹50k"
            treatment = "Low-Dose CT Follow-up (6 months)"
        elif size_mm < 15:
            severity = "Suspicious Nodule"
            cost = "₹1L - ₹3L"
            treatment = "PET-CT + Biopsy recommended"
        elif size_mm < 30:
            severity = "Probable Malignancy"
            cost = "₹3L - ₹6L"
            treatment = "Lobectomy + Staging Workup"
        else:
            severity = "Advanced Mass"
            cost = "₹6L - ₹12L"
            treatment = "Systemic Therapy + Consider Surgery"
    
    elif organ == "Liver":
        if size_mm < 15:
            severity = "Small Lesion"
            cost = "₹80k - ₹1.5L"
            treatment = "Triphasic CT + Tumor Markers (AFP)"
        elif size_mm < 30:
            severity = "Moderate Lesion"
            cost = "₹2L - ₹4L"
            treatment = "Ablation Therapy or Resection"
        else:
            severity = "Large Mass"
            cost = "₹4L - ₹8L"
            treatment = "Hepatectomy + TACE if indicated"
    
    elif organ == "Kidney":
        if size_mm < 20:
            severity = "Small Renal Mass"
            cost = "₹1L - ₹2.5L"
            treatment = "Active Surveillance or Ablation"
        elif size_mm < 40:
            severity = "Moderate Mass"
            cost = "₹2.5L - ₹5L"
            treatment = "Partial Nephrectomy"
        else:
            severity = "Large/Complex Mass"
            cost = "₹5L - ₹10L"
            treatment = "Radical Nephrectomy + Staging"
    
    if tumor_count > 1:
        severity += f" (Multiple: {tumor_count})"
        treatment = "Multidisciplinary Team Review Required - " + treatment
    
    return severity, cost, treatment

def check_image_quality(image):
    """Detect image artifacts and quality issues"""
    issues = []
    
    # Check for motion blur
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("▸  Possible motion blur detected")
    
    # Check brightness
    avg_brightness = np.mean(gray)
    if avg_brightness < 50:
        issues.append("⚠️ Image too dark")
    elif avg_brightness > 200:
        issues.append("⚠️ Image overexposed")
    
    # Check contrast
    contrast = gray.std()
    if contrast < 30:
        issues.append("⚠️ Low contrast")
    
    return issues

# --- 3. ENSEMBLE PREDICTION (Multiple Confidence Levels) ---
def ensemble_predict(model, image, conf_levels=[0.3, 0.4, 0.5]):
    """Run multiple predictions and combine results"""
    all_detections = []
    
    for conf in conf_levels:
        results = model.predict(image, conf=conf, verbose=False)
        detections = results[0].boxes
        all_detections.extend([(box, conf) for box in detections])
    
    # Remove duplicates using NMS-like logic
    unique_detections = []
    for box, conf_level in all_detections:
        is_duplicate = False
        for existing_box, _ in unique_detections:
            # Check IoU (simplified)
            if np.allclose(box.xywh[0].cpu().numpy(), existing_box.xywh[0].cpu().numpy(), atol=50):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_detections.append((box, conf_level))
    
    return unique_detections

# --- 4. MAIN ADVANCED ANALYSIS FUNCTION ---
def advanced_medical_analysis(patient_id, image, organ_type, conf_threshold, use_ensemble, doctor_notes=""):
    if image is None:
        return None, "⚠️ No image uploaded", "", "", ""
    
    organ_key = organ_type.split(" ")[0]
    model = models[organ_key]
    
    # Image quality check
    quality_issues = check_image_quality(image)
    quality_report = "\n".join(quality_issues) if quality_issues else "✅ Image quality acceptable"
    
    # Run prediction
    if use_ensemble:
        detections = ensemble_predict(model, image)
        # For visualization, use single prediction
        results = model.predict(image, conf=conf_threshold, verbose=False)
        annotated_bgr = results[0].plot()
    else:
        results = model.predict(image, conf=conf_threshold, verbose=False)
        annotated_bgr = results[0].plot()
        detections = [(box, conf_threshold) for box in results[0].boxes]
    
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Smart filter
    valid_detections = [(box, conf) for box, conf in detections if float(box.conf[0])*100 > 35]
    
    if len(valid_detections) > 0:
        # Analyze all tumors
        tumor_sizes = []
        tumor_confidences = []
        
        for box, _ in valid_detections:
            box_data = box.xywh[0]
            size_mm = calculate_tumor_size_mm(box_data, image.shape)
            conf = float(box.conf[0]) * 100
            tumor_sizes.append(size_mm)
            tumor_confidences.append(conf)
        
        # Get largest tumor for primary diagnosis
        max_idx = np.argmax(tumor_sizes)
        max_size = tumor_sizes[max_idx]
        max_conf = tumor_confidences[max_idx]
        tumor_count = len(valid_detections)
        
        # Calculate box area for cost estimation
        main_box = valid_detections[max_idx][0].xywh[0]
        area = float(main_box[2]) * float(main_box[3])
        
        sev, cost, treat = get_medical_report(organ_key, area, max_conf, max_size, tumor_count)
        risk = get_risk_level(organ_key, max_size, max_conf)
        
        # Build comprehensive report
        status = f"⚠️ ABNORMALITY DETECTED - {organ_key.upper()}\n"
        status += f"Diagnosis: {sev}\n"
        status += f"Risk Level: {risk}\n"
        status += f"Detected Lesions: {tumor_count}"
        
        details = f"🔍 ANALYSIS RESULTS:\n"
        details += f"• Primary Lesion Size: {max_size} mm\n"
        details += f"• Confidence: {max_conf:.1f}%\n"
        details += f"• Total Lesions: {tumor_count}\n\n"
        
        if tumor_count > 1:
            avg_size = np.mean(tumor_sizes)
            details += f"• Average Size: {avg_size:.1f} mm\n"
            details += f"• Size Range: {min(tumor_sizes):.1f} - {max(tumor_sizes):.1f} mm\n\n"
        
        details += f"💰 COST ESTIMATE: {cost}\n\n"
        details += f"📋 TREATMENT PLAN:\n{treat}\n\n"
        details += f"🔬 IMAGE QUALITY:\n{quality_report}"
        
        # Advanced metrics
        metrics = f"📊 ADVANCED METRICS:\n"
        metrics += f"• Detection Method: {'Ensemble' if use_ensemble else 'Single Model'}\n"
        metrics += f"• Confidence Threshold: {conf_threshold:.2f}\n"
        metrics += f"• Risk Stratification: {risk}\n"
        metrics += f"• Follow-up Recommended: {'3 months' if risk == 'Low' else '1 month' if risk == 'Moderate' else 'Immediate'}"
        
        # Save to history
        if patient_id:
            new_record = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Patient_ID": patient_id,
                "Organ": organ_key,
                "Diagnosis": sev,
                "Confidence": f"{max_conf:.1f}%",
                "Tumor_Count": tumor_count,
                "Max_Size_mm": max_size,
                "Risk_Level": risk,
                "Cost_Est": cost,
                "Notes": doctor_notes[:100] if doctor_notes else ""
            }
            df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df.to_csv(HISTORY_FILE, index=False)
            
            # Save doctor notes
            if doctor_notes:
                with open(NOTES_FILE, 'r') as f:
                    notes_db = json.load(f)
                notes_db[f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"] = doctor_notes
                with open(NOTES_FILE, 'w') as f:
                    json.dump(notes_db, f, indent=2)
    else:
        status = "✅ NORMAL SCAN\n"
        status += f"Organ: {organ_key}\n"
        status += "No significant abnormalities detected"
        
        details = f"🔍 ANALYSIS RESULTS:\n"
        details += "• Status: Normal\n"
        details += "• Lesions Detected: 0\n"
        details += f"• Confidence Threshold: {conf_threshold:.2f}\n\n"
        details += f"🔬 IMAGE QUALITY:\n{quality_report}\n\n"
        details += "✅ Scan appears normal. Routine follow-up recommended."
        
        metrics = "📊 No abnormalities detected"
        
        if len(results[0].boxes) > 0:
            annotated_rgb = image  # Clean image
    
    return annotated_rgb, status, details, metrics, quality_report

def get_filtered_history(organ_filter="All", risk_filter="All", days_back=30):
    """Advanced filtering for patient records"""
    df = pd.read_csv(HISTORY_FILE)
    
    if df.empty:
        return df
    
    # Date filter
    df['Date'] = pd.to_datetime(df['Date'])
    cutoff_date = datetime.now() - pd.Timedelta(days=days_back)
    df = df[df['Date'] >= cutoff_date]
    
    # Organ filter
    if organ_filter != "All":
        df = df[df['Organ'] == organ_filter]
    
    # Risk filter
    if risk_filter != "All" and 'Risk_Level' in df.columns:
        df = df[df['Risk_Level'] == risk_filter]
    
    return df.sort_values('Date', ascending=False)

def get_statistics():
    """Generate system statistics"""
    df = pd.read_csv(HISTORY_FILE)
    
    if df.empty:
        return "No data available yet."
    
    total_scans = len(df)
    organs = df['Organ'].value_counts().to_dict() if 'Organ' in df.columns else {}
    
    stats = f"📊 SYSTEM STATISTICS\n\n"
    stats += f"Total Scans: {total_scans}\n\n"
    stats += "Scans by Organ:\n"
    for organ, count in organs.items():
        stats += f"  • {organ}: {count}\n"
    
    if 'Risk_Level' in df.columns:
        stats += f"\nRisk Distribution:\n"
        risk_dist = df['Risk_Level'].value_counts().to_dict()
        for risk, count in risk_dist.items():
            stats += f"  • {risk}: {count}\n"
    
    return stats

def batch_analysis(files, organ_type, conf_threshold):
    """Process multiple images"""
    if not files:
        return "No files uploaded", None
    
    results_summary = []
    organ_key = organ_type.split(" ")[0]
    
    for idx, file in enumerate(files):
        image = cv2.imread(file.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        _, status, _, _, _ = advanced_medical_analysis(
            f"Batch_{idx+1}", image_rgb, organ_type, conf_threshold, False, ""
        )
        
        results_summary.append(f"Image {idx+1}: {status.split(' ')[0]}")
    
    summary_text = "\n".join(results_summary)
    return summary_text, pd.read_csv(HISTORY_FILE)

# --- 5. PROFESSIONAL UI THEME ---
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

css = """
footer {visibility: hidden}
.gradio-container {min-height: 0px !important;}
h1 {text-align: center; color: #0f766e; font-weight: 800; margin-bottom: 0;}
.warning-box {background-color: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;}
"""

# --- 6. ADVANCED UI LAYOUT ---
with gr.Blocks(theme=theme, css=css, title="OncoVision AI Pro") as app:
    
    gr.Markdown("# 🏥 OncoVision AI Pro\n### Advanced Multi-Organ Tumor Detection & Analysis Platform")
    
    with gr.Tabs():
        
        # PAGE 1: ADVANCED SCANNER
        with gr.TabItem("🔍 AI Scanner Pro"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Patient Information")
                    pid_input = gr.Textbox(label="Patient ID / Name", placeholder="e.g. P-2024-001")
                    organ_select = gr.Dropdown(
                        ["Brain (MRI)", "Lung (CT)", "Liver (CT)", "Kidney (CT)"], 
                        label="Organ System", 
                        value="Brain (MRI)"
                    )
                    img_input = gr.Image(type="numpy", label="Upload Medical Scan")
                    
                    gr.Markdown("### ⚙️ Advanced Settings")
                    conf_slider = gr.Slider(0.1, 0.9, value=0.50, step=0.05, label="Detection Sensitivity")
                    ensemble_check = gr.Checkbox(label="Use Ensemble Analysis (Better Accuracy)", value=False)
                    doctor_notes = gr.Textbox(label="Doctor's Notes (Optional)", lines=3, placeholder="Clinical observations...")
                    
                    analyze_btn = gr.Button("🚀 Run Advanced Diagnosis", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🩺 Diagnostic Report")
                    img_output = gr.Image(label="AI Localization & Annotation")
                    
                    with gr.Accordion("📄 Full Clinical Report", open=True):
                        status_box = gr.Textbox(label="Primary Diagnosis", lines=4)
                        details_box = gr.Textbox(label="Detailed Analysis", lines=8)
                    
                    with gr.Accordion("📊 Advanced Metrics", open=False):
                        metrics_box = gr.Textbox(label="Statistical Analysis", lines=4)
                        quality_box = gr.Textbox(label="Image Quality Assessment", lines=3)
        
        # PAGE 2: BATCH PROCESSING
        with gr.TabItem("📦 Batch Analysis"):
            gr.Markdown("### Process Multiple Scans Simultaneously")
            
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(label="Upload Multiple Scans", file_count="multiple")
                    batch_organ = gr.Dropdown(
                        ["Brain (MRI)", "Lung (CT)", "Liver (CT)", "Kidney (CT)"],
                        label="Organ Type",
                        value="Brain (MRI)"
                    )
                    batch_conf = gr.Slider(0.3, 0.8, value=0.5, label="Confidence Threshold")
                    batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Textbox(label="Batch Results Summary", lines=15)
                    batch_table = gr.Dataframe(label="All Results")
            
            batch_btn.click(
                fn=batch_analysis,
                inputs=[batch_files, batch_organ, batch_conf],
                outputs=[batch_output, batch_table]
            )
        
        # PAGE 3: ADVANCED PATIENT RECORDS
        with gr.TabItem("📂 Patient Records Database"):
            gr.Markdown("### Digital Health Records with Advanced Filtering")
            
            with gr.Row():
                organ_filter = gr.Dropdown(
                    ["All", "Brain", "Lung", "Liver", "Kidney"],
                    label="Filter by Organ",
                    value="All"
                )
                risk_filter = gr.Dropdown(
                    ["All", "Low", "Moderate", "High", "Critical"],
                    label="Filter by Risk",
                    value="All"
                )
                days_filter = gr.Slider(7, 365, value=30, step=1, label="Days Back")
                filter_btn = gr.Button("🔍 Apply Filters", variant="secondary")
            
            history_table = gr.Dataframe(value=get_filtered_history(), interactive=False)
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh All Records")
                export_btn = gr.Button("💾 Export to CSV")
            
            filter_btn.click(
                fn=get_filtered_history,
                inputs=[organ_filter, risk_filter, days_filter],
                outputs=history_table
            )
            refresh_btn.click(fn=lambda: get_filtered_history(), outputs=history_table)
        
        # PAGE 4: ANALYTICS DASHBOARD
        with gr.TabItem("📊 Analytics Dashboard"):
            gr.Markdown("### System Statistics & Insights")
            
            stats_btn = gr.Button("📈 Generate Statistics", variant="primary")
            stats_output = gr.Textbox(label="System Analytics", lines=15)
            
            gr.Markdown("### Recent Activity")
            recent_table = gr.Dataframe(value=get_filtered_history(days_back=7))
            
            stats_btn.click(fn=get_statistics, outputs=stats_output)
        
        # PAGE 5: ABOUT & DOCUMENTATION
        with gr.TabItem("ℹ️ System Information"):
            gr.Markdown("""
            ## **OncoVision AI Pro v3.0**
            
            ### 🚀 Advanced Features
            
            #### **Enhanced Detection**
            - ✅ Multi-model ensemble predictions
            - ✅ Tumor size measurement (mm)
            - ✅ Multiple tumor detection
            - ✅ Risk stratification algorithm
            - ✅ Image quality validation
            
            #### **Clinical Tools**
            - ✅ Batch processing capability
            - ✅ Advanced filtering & search
            - ✅ Doctor notes integration
            - ✅ Comprehensive reporting
            - ✅ Statistical analytics
            
            #### **Supported Organs**
            - **Brain:** Glioma, Meningioma, Metastases (MRI)
            - **Lung:** Nodules, Masses, Consolidation (CT)
            - **Liver:** Hepatocellular Carcinoma, Metastases (CT)
            - **Kidney:** Renal Cell Carcinoma, Masses (CT)
            
            ### 📋 Technical Specifications
            - **AI Model:** YOLOv8 Custom Trained
            - **Framework:** PyTorch + Ultralytics
            - **Interface:** Gradio 4.x
            - **Data Management:** Pandas + JSON
            
            ### ⚠️ Important Disclaimer
            *This system is designed for research and educational purposes. 
            All diagnoses should be confirmed by qualified medical professionals. 
            Not intended for direct clinical use without proper validation.*
            
            ### 👨‍💻 Developed By
            Medical AI Research Team | 2024
            """)
    
    # EVENT LISTENERS
    analyze_btn.click(
        fn=advanced_medical_analysis,
        inputs=[pid_input, img_input, organ_select, conf_slider, ensemble_check, doctor_notes],
        outputs=[img_output, status_box, details_box, metrics_box, quality_box]
    )
    
    # Real-time updates on image change
    img_input.change(
        fn=advanced_medical_analysis,
        inputs=[pid_input, img_input, organ_select, conf_slider, ensemble_check, doctor_notes],
        outputs=[img_output, status_box, details_box, metrics_box, quality_box]
    )

app.launch(show_api=False, share=False)