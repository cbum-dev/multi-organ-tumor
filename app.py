import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- 1. SETUP & MODELS ---
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

# CSV File for History
HISTORY_FILE = "patient_records.csv"
if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(columns=["Date", "Patient_ID", "Organ", "Diagnosis", "Confidence", "Cost_Est"])
    df.to_csv(HISTORY_FILE, index=False)

# --- 2. MEDICAL LOGIC ---
def get_medical_report(organ, box_area, confidence):
    severity, cost, treatment = "Unknown", "N/A", "N/A"
    
    if organ == "Brain":
        if box_area < 3000:
            severity = "Stage I (Early)"
            cost = "₹1.5L - ₹3L"
            treatment = "Observation / Minimally Invasive"
        elif box_area < 8000:
            severity = "Stage II (Moderate)"
            cost = "₹3L - ₹6L"
            treatment = "Surgical Resection"
        else:
            severity = "Stage III/IV (Critical)"
            cost = "₹8L+"
            treatment = "Urgent Craniotomy + Radiation"
    
    # (Add other organ logic here as per previous code)
    # ... using simplified defaults for demo:
    elif organ == "Lung":
        severity = "Nodule (Suspicious)" if box_area < 2000 else "Mass (High Risk)"
        cost = "₹50k" if box_area < 2000 else "₹3L+"
        treatment = "Follow-up CT" if box_area < 2000 else "Biopsy + Surgery"
    
    return severity, cost, treatment

# --- 3. MAIN ANALYSIS FUNCTION ---
def medical_analysis(patient_id, image, organ_type, conf_threshold):
    organ_key = organ_type.split(" ")[0]
    model = models[organ_key]
    
    results = model.predict(image, conf=conf_threshold)
    
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    detections = results[0].boxes
    
    # Smart Filter (Ignore noise < 35%)
    valid_detections = [box for box in detections if float(box.conf[0])*100 > 35]

    if len(valid_detections) > 0:
        box = valid_detections[0].xywh[0]
        area = float(box[2]) * float(box[3])
        conf = float(valid_detections[0].conf[0]) * 100
        
        sev, cost, treat = get_medical_report(organ_key, area, conf)

        status = f"⚠️ {organ_key} Tumor Detected\nSeverity: {sev}"
        details = f"Confidence: {conf:.1f}%\nEst. Cost: {cost}\nPlan: {treat}"
        
        # SAVE TO HISTORY
        if patient_id:
            new_record = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Patient_ID": patient_id,
                "Organ": organ_key,
                "Diagnosis": sev,
                "Confidence": f"{conf:.1f}%",
                "Cost_Est": cost
            }
            df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df.to_csv(HISTORY_FILE, index=False)
            
    else:
        status = "✅ Normal Scan"
        details = "No significant anomalies detected."
        if len(detections) > 0: annotated_rgb = image # Clean noise

    return annotated_rgb, status, details

def get_history():
    return pd.read_csv(HISTORY_FILE)

# --- 4. PRO UI THEME ---
# Using 'Soft' theme with Teal (Medical) primary color
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
"""

# --- 5. UI LAYOUT (MULTI-PAGE) ---
with gr.Blocks(theme=theme, css=css, title="OncoVision AI") as app:
    
    # Header
    with gr.Row():
        gr.Markdown("#OncoVision AI Platform\n### Next-Gen Multi-Organ Tumor Detection System")

    # NAVIGATION TABS
    with gr.Tabs():
        
        # PAGE 1: THE SCANNER
        with gr.TabItem("🔍 AI Scanner"):
            with gr.Row():
                # Left: Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Patient & Config")
                    pid_input = gr.Textbox(label="Patient ID / Name", placeholder="e.g. John Doe")
                    organ_select = gr.Dropdown(["Brain (MRI)", "Lung (CT)", "Liver (CT)", "Kidney (CT)"], label="Organ System", value="Brain (MRI)")
                    img_input = gr.Image(type="numpy", label="Upload Medical Scan")
                    conf_slider = gr.Slider(0.1, 1.0, value=0.50, label="AI Sensitivity")
                    
                    analyze_btn = gr.Button("🚀 Run Diagnosis", variant="primary")

                # Right: Report
                with gr.Column(scale=1):
                    gr.Markdown("### 🩺 Diagnostic Results")
                    img_output = gr.Image(label="AI Localization")
                    with gr.Group():
                        status_box = gr.Textbox(label="Clinical Status", lines=2)
                        details_box = gr.Textbox(label="Treatment Plan", lines=4)

        # PAGE 2: PATIENT HISTORY
        with gr.TabItem("📂 Patient Records"):
            gr.Markdown("### Digital Health Records Database")
            refresh_btn = gr.Button("🔄 Refresh Records")
            history_table = gr.Dataframe(value=get_history(), interactive=False)
            
            refresh_btn.click(fn=get_history, inputs=None, outputs=history_table)

        # PAGE 3: ABOUT
        with gr.TabItem("ℹ️ About System"):
            gr.Markdown("""
            ### **OncoVision AI v2.0**
            **Developed by:** [Your Name] & Team  
            **Tech Stack:** YOLOv8, PyTorch, Gradio, Pandas  
            
            #### **Capabilities:**
            * **Brain:** Glioma/Meningioma Detection (MRI)
            * **Lung:** Nodule/Mass Detection (CT)
            * **Liver & Kidney:** Lesion Localization
            
            *For Research Use Only. Not for Clinical Diagnosis.*
            """)

    # EVENT LISTENERS (Real-time)
    conf_slider.change(fn=medical_analysis, inputs=[pid_input, img_input, organ_select, conf_slider], outputs=[img_output, status_box, details_box])
    organ_select.change(fn=medical_analysis, inputs=[pid_input, img_input, organ_select, conf_slider], outputs=[img_output, status_box, details_box])
    img_input.change(fn=medical_analysis, inputs=[pid_input, img_input, organ_select, conf_slider], outputs=[img_output, status_box, details_box])
    analyze_btn.click(fn=medical_analysis, inputs=[pid_input, img_input, organ_select, conf_slider], outputs=[img_output, status_box, details_box])

app.launch(show_api=False)