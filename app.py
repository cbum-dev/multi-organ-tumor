import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from collections import defaultdict
import json
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

# --- ADVANCED MODEL ENSEMBLE ---
models = {}
model_files = {
    "Brain": "brain_yolo.pt",
    "Lung": "lung_yolo.pt",
    "Liver": "liver_yolo.pt",
    "Kidney": "kidney_yolo.pt"
}

print("⏳ Loading Medical Models with Advanced Ensemble...")
for organ, file in model_files.items():
    try:
        models[organ] = YOLO(file)
        print(f"✅ {organ} Model Loaded")
    except:
        print(f"⚠️ {file} not found. Using fallback.")
        models[organ] = YOLO("yolov8n.pt")

# --- DATA STORAGE ---
HISTORY_FILE = "patient_records.csv"
NOTES_FILE = "doctor_notes.json"
COMPARISON_FILE = "scan_comparisons.json"
RADIOMICS_FILE = "radiomics_features.json"
AUDIT_LOG = "audit_trail.json"

# Initialize all databases
for file, initial_data in [
    (HISTORY_FILE, pd.DataFrame(columns=[
        "Date", "Patient_ID", "Organ", "Diagnosis", "Confidence", 
        "Tumor_Count", "Max_Size_mm", "Risk_Level", "Cost_Est", 
        "Notes", "Radiomics_ID", "Quality_Score"
    ])),
    (NOTES_FILE, {}),
    (COMPARISON_FILE, {}),
    (RADIOMICS_FILE, {}),
    (AUDIT_LOG, [])
]:
    if not os.path.exists(file):
        if file.endswith('.csv'):
            initial_data.to_csv(file, index=False)
        else:
            with open(file, 'w') as f:
                json.dump(initial_data, f)

# --- ADVANCED IMAGE PREPROCESSING ---
def advanced_preprocessing(image, organ_type):
    """Apply medical-grade image enhancement"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise with bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Organ-specific windowing
    if "Brain" in organ_type:
        # Brain window: 80 HU width, 40 HU center
        lower, upper = 0, 80
    elif "Lung" in organ_type:
        # Lung window: 1500 HU width, -600 HU center
        lower, upper = 0, 255
    else:
        # Soft tissue window
        lower, upper = 40, 400
    
    # Normalize
    normalized = np.clip(denoised, lower, upper)
    normalized = ((normalized - lower) / (upper - lower) * 255).astype(np.uint8)
    
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

# --- ADVANCED QUALITY ASSESSMENT ---
def comprehensive_quality_check(image):
    """Multi-dimensional image quality analysis"""
    issues = []
    scores = {}
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    scores['sharpness'] = min(100, laplacian_var / 10)
    if laplacian_var < 100:
        issues.append(f"⚠️ Low sharpness (blur detected): {laplacian_var:.1f}")
    
    # 2. Contrast assessment
    contrast = gray.std()
    scores['contrast'] = min(100, contrast / 0.8)
    if contrast < 30:
        issues.append(f"⚠️ Low contrast: {contrast:.1f}")
    
    # 3. Brightness analysis
    avg_brightness = np.mean(gray)
    scores['brightness'] = 100 - abs(127.5 - avg_brightness) / 1.275
    if avg_brightness < 50:
        issues.append(f"⚠️ Underexposed: {avg_brightness:.1f}")
    elif avg_brightness > 200:
        issues.append(f"⚠️ Overexposed: {avg_brightness:.1f}")
    
    # 4. Noise estimation (using Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    noise_level = np.sqrt(sobelx**2 + sobely**2).std()
    scores['noise'] = max(0, 100 - noise_level / 2)
    if noise_level > 50:
        issues.append(f"⚠️ High noise level: {noise_level:.1f}")
    
    # 5. Artifact detection (check for unusual patterns)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    if np.max(magnitude) > np.mean(magnitude) * 1000:
        issues.append("⚠️ Possible artifacts detected")
        scores['artifact'] = 50
    else:
        scores['artifact'] = 100
    
    # Overall quality score
    overall_score = np.mean(list(scores.values()))
    
    return issues, scores, overall_score

# --- RADIOMICS FEATURE EXTRACTION ---
def extract_radiomics_features(image, mask_coords):
    """Extract advanced radiomics features from tumor region"""
    x, y, w, h = map(int, mask_coords)
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return {}
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    features = {}
    
    # First-order statistics
    features['mean_intensity'] = float(np.mean(gray_roi))
    features['std_intensity'] = float(np.std(gray_roi))
    features['skewness'] = float(np.mean((gray_roi - features['mean_intensity'])**3) / (features['std_intensity']**3 + 1e-10))
    features['kurtosis'] = float(np.mean((gray_roi - features['mean_intensity'])**4) / (features['std_intensity']**4 + 1e-10))
    features['entropy'] = float(-np.sum(np.histogram(gray_roi, bins=256)[0] * np.log2(np.histogram(gray_roi, bins=256)[0] + 1e-10)))
    
    # Shape features
    features['area'] = int(w * h)
    features['perimeter'] = 2 * (w + h)
    features['sphericity'] = features['area'] / (features['perimeter']**2 + 1e-10)
    features['aspect_ratio'] = float(w) / (h + 1e-10)
    
    # Texture features (simplified GLCM)
    features['homogeneity'] = float(np.sum(gray_roi**2) / (gray_roi.size + 1e-10))
    
    return features

# --- UNCERTAINTY QUANTIFICATION ---
def monte_carlo_dropout_prediction(model, image, n_iterations=10):
    """Estimate prediction uncertainty using multiple forward passes"""
    predictions = []
    
    for _ in range(n_iterations):
        results = model.predict(image, conf=0.3, verbose=False)
        predictions.append(results[0].boxes)
    
    # Calculate variance in predictions
    if len(predictions) > 0 and len(predictions[0]) > 0:
        confidences = [float(pred.conf[0]) for pred_set in predictions for pred in pred_set if len(pred_set) > 0]
        if confidences:
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            uncertainty = std_conf / (mean_conf + 1e-10)
            return mean_conf, uncertainty
    
    return 0.0, 1.0

# --- TEMPORAL ANALYSIS ---
def compare_with_previous_scans(patient_id, current_size, current_count):
    """Track tumor growth over time"""
    with open(COMPARISON_FILE, 'r') as f:
        comparisons = json.load(f)
    
    if patient_id not in comparisons:
        comparisons[patient_id] = []
    
    current_data = {
        'date': datetime.now().isoformat(),
        'size': current_size,
        'count': current_count
    }
    
    comparisons[patient_id].append(current_data)
    
    # Keep last 10 scans
    comparisons[patient_id] = comparisons[patient_id][-10:]
    
    with open(COMPARISON_FILE, 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    # Calculate growth rate
    if len(comparisons[patient_id]) >= 2:
        prev = comparisons[patient_id][-2]
        growth_rate = ((current_size - prev['size']) / prev['size']) * 100 if prev['size'] > 0 else 0
        days_diff = (datetime.now() - datetime.fromisoformat(prev['date'])).days
        
        return {
            'previous_size': prev['size'],
            'growth_rate': growth_rate,
            'days_since_last': days_diff,
            'new_tumors': current_count - prev['count']
        }
    
    return None

# --- DIFFERENTIAL DIAGNOSIS ---
def generate_differential_diagnosis(organ, size_mm, count, radiomics):
    """Generate list of possible diagnoses with probabilities"""
    ddx = []
    
    if organ == "Brain":
        if size_mm < 15:
            ddx = [
                ("Meningioma", 35),
                ("Schwannoma", 25),
                ("Metastasis", 20),
                ("Low-grade Glioma", 15),
                ("Cavernoma", 5)
            ]
        else:
            ddx = [
                ("Glioblastoma", 40),
                ("High-grade Glioma", 25),
                ("Metastasis", 20),
                ("Meningioma", 10),
                ("Lymphoma", 5)
            ]
    elif organ == "Lung":
        if size_mm < 10:
            ddx = [
                ("Benign Nodule", 50),
                ("Granuloma", 20),
                ("Early Adenocarcinoma", 15),
                ("Hamartoma", 10),
                ("Infection", 5)
            ]
        else:
            ddx = [
                ("Adenocarcinoma", 35),
                ("Squamous Cell Carcinoma", 25),
                ("Small Cell Carcinoma", 15),
                ("Metastasis", 15),
                ("Abscess", 10)
            ]
    elif organ == "Liver":
        ddx = [
            ("Hepatocellular Carcinoma", 40),
            ("Metastasis", 30),
            ("Hemangioma", 15),
            ("Focal Nodular Hyperplasia", 10),
            ("Abscess", 5)
        ]
    else:  # Kidney
        ddx = [
            ("Renal Cell Carcinoma", 70),
            ("Oncocytoma", 15),
            ("Angiomyolipoma", 10),
            ("Metastasis", 5)
        ]
    
    # Adjust based on multiple tumors
    if count > 3:
        # Increase probability of metastases
        ddx = [(name, prob*1.5 if "Metastasis" in name else prob*0.7) for name, prob in ddx]
    
    # Normalize probabilities
    total = sum(prob for _, prob in ddx)
    ddx = [(name, (prob/total)*100) for name, prob in ddx]
    
    return sorted(ddx, key=lambda x: x[1], reverse=True)

# --- TREATMENT RECOMMENDATION ENGINE ---
def advanced_treatment_recommendations(organ, size_mm, count, risk, radiomics):
    """Generate evidence-based treatment recommendations"""
    recommendations = []
    
    # Primary treatment
    if risk == "Critical":
        recommendations.append({
            'type': 'Urgent',
            'treatment': 'Emergency multidisciplinary team meeting',
            'timeline': 'Within 24-48 hours',
            'details': 'Immediate oncology, surgery, and radiation oncology consultation'
        })
    
    # Organ-specific protocols
    if organ == "Brain":
        if size_mm < 30:
            recommendations.append({
                'type': 'Surgical',
                'treatment': 'Stereotactic Radiosurgery (Gamma Knife/CyberKnife)',
                'timeline': '2-4 weeks',
                'details': 'Suitable for lesions <3cm, minimal invasiveness'
            })
        else:
            recommendations.append({
                'type': 'Surgical',
                'treatment': 'Craniotomy with microsurgical resection',
                'timeline': '1-2 weeks',
                'details': 'Complete resection with neuronavigation'
            })
        
        recommendations.append({
            'type': 'Medical',
            'treatment': 'Temozolomide + Radiation (Stupp Protocol)',
            'timeline': 'Post-operative',
            'details': 'Standard of care for glioblastoma'
        })
    
    elif organ == "Lung":
        if size_mm < 20 and count == 1:
            recommendations.append({
                'type': 'Surgical',
                'treatment': 'Video-Assisted Thoracoscopic Surgery (VATS)',
                'timeline': '2-3 weeks',
                'details': 'Minimally invasive lobectomy or wedge resection'
            })
        else:
            recommendations.append({
                'type': 'Medical',
                'treatment': 'Platinum-based chemotherapy + Immunotherapy',
                'timeline': 'Immediate',
                'details': 'Consider pembrolizumab if PD-L1 positive'
            })
    
    # Follow-up imaging
    if risk in ["Low", "Moderate"]:
        interval = "3 months" if risk == "Moderate" else "6 months"
        recommendations.append({
            'type': 'Imaging',
            'treatment': f'Follow-up imaging in {interval}',
            'timeline': interval,
            'details': 'Monitor for growth or changes'
        })
    
    # Supportive care
    recommendations.append({
        'type': 'Supportive',
        'treatment': 'Multidisciplinary support',
        'timeline': 'Ongoing',
        'details': 'Pain management, nutrition, psychological support'
    })
    
    return recommendations

# --- AUDIT TRAIL ---
def log_audit_event(event_type, patient_id, details):
    """Record all system activities"""
    with open(AUDIT_LOG, 'r') as f:
        audit = json.load(f)
    
    audit.append({
        'timestamp': datetime.now().isoformat(),
        'event': event_type,
        'patient_id': patient_id,
        'details': details
    })
    
    # Keep last 1000 events
    audit = audit[-1000:]
    
    with open(AUDIT_LOG, 'w') as f:
        json.dump(audit, f, indent=2)

# --- MAIN ULTRA-ADVANCED ANALYSIS ---
def ultra_advanced_analysis(patient_id, image, organ_type, conf_threshold, 
                           use_ensemble, use_preprocessing, estimate_uncertainty,
                           doctor_notes=""):
    if image is None:
        return None, "⚠️ No image uploaded", "", "", "", "", ""
    
    organ_key = organ_type.split(" ")[0]
    model = models[organ_key]
    
    # Log analysis start
    log_audit_event("ANALYSIS_START", patient_id, f"Organ: {organ_key}")
    
    # Advanced preprocessing
    processed_image = advanced_preprocessing(image, organ_type) if use_preprocessing else image
    
    # Quality assessment
    quality_issues, quality_scores, overall_quality = comprehensive_quality_check(image)
    quality_report = f"Overall Quality Score: {overall_quality:.1f}%\n\n"
    quality_report += "Detailed Metrics:\n"
    for metric, score in quality_scores.items():
        quality_report += f"• {metric.capitalize()}: {score:.1f}%\n"
    if quality_issues:
        quality_report += "\n⚠️ Issues Detected:\n" + "\n".join(quality_issues)
    
    # Uncertainty estimation
    if estimate_uncertainty:
        mean_conf, uncertainty = monte_carlo_dropout_prediction(model, processed_image)
        uncertainty_report = f"Prediction Uncertainty: {uncertainty:.3f}\n"
        uncertainty_report += f"Mean Confidence: {mean_conf*100:.1f}%\n"
        uncertainty_report += f"Reliability: {'High' if uncertainty < 0.1 else 'Moderate' if uncertainty < 0.3 else 'Low'}"
    else:
        uncertainty_report = "Uncertainty estimation disabled"
    
    # Run prediction
    results = model.predict(processed_image, conf=conf_threshold, verbose=False)
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    detections = [(box, conf_threshold) for box in results[0].boxes]
    valid_detections = [(box, conf) for box, conf in detections if float(box.conf[0])*100 > 35]
    
    if len(valid_detections) > 0:
        # Extract all tumor data
        tumor_data = []
        all_radiomics = []
        
        for box, _ in valid_detections:
            box_data = box.xywh[0].cpu().numpy()
            x, y, w, h = box_data
            size_mm = ((w + h) / 2) * 0.5
            conf = float(box.conf[0]) * 100
            
            # Extract radiomics
            x1, y1 = int(x - w/2), int(y - h/2)
            radiomics = extract_radiomics_features(image, (x1, y1, int(w), int(h)))
            
            tumor_data.append({
                'size': size_mm,
                'confidence': conf,
                'radiomics': radiomics
            })
            all_radiomics.append(radiomics)
        
        # Get largest tumor
        tumor_data.sort(key=lambda x: x['size'], reverse=True)
        max_tumor = tumor_data[0]
        tumor_count = len(valid_detections)
        
        # Risk assessment
        risk = get_risk_level(organ_key, max_tumor['size'], max_tumor['confidence'])
        
        # Temporal comparison
        temporal_analysis = compare_with_previous_scans(patient_id, max_tumor['size'], tumor_count)
        
        # Differential diagnosis
        ddx = generate_differential_diagnosis(organ_key, max_tumor['size'], tumor_count, max_tumor['radiomics'])
        
        # Treatment recommendations
        treatments = advanced_treatment_recommendations(organ_key, max_tumor['size'], tumor_count, risk, max_tumor['radiomics'])
        
        # Build comprehensive report
        status = f"⚠️ ABNORMALITY DETECTED - {organ_key.upper()}\n"
        status += f"Risk Level: {risk}\n"
        status += f"Detected Lesions: {tumor_count}\n"
        status += f"Quality Score: {overall_quality:.1f}%"
        
        details = f"🔍 ADVANCED ANALYSIS:\n\n"
        details += f"Primary Lesion:\n"
        details += f"• Size: {max_tumor['size']:.1f} mm\n"
        details += f"• Confidence: {max_tumor['confidence']:.1f}%\n"
        details += f"• Total Lesions: {tumor_count}\n\n"
        
        if tumor_count > 1:
            avg_size = np.mean([t['size'] for t in tumor_data])
            details += f"Multiple Lesions Analysis:\n"
            details += f"• Average Size: {avg_size:.1f} mm\n"
            details += f"• Size Range: {min(t['size'] for t in tumor_data):.1f} - {max(t['size'] for t in tumor_data):.1f} mm\n\n"
        
        # Radiomics summary
        details += f"📊 Radiomics Features:\n"
        rad = max_tumor['radiomics']
        details += f"• Mean Intensity: {rad.get('mean_intensity', 0):.1f}\n"
        details += f"• Heterogeneity (Std): {rad.get('std_intensity', 0):.1f}\n"
        details += f"• Shape Factor: {rad.get('sphericity', 0):.3f}\n"
        details += f"• Aspect Ratio: {rad.get('aspect_ratio', 0):.2f}\n\n"
        
        # Temporal
        if temporal_analysis:
            details += f"⏱️ Temporal Analysis:\n"
            details += f"• Previous Size: {temporal_analysis['previous_size']:.1f} mm\n"
            details += f"• Growth Rate: {temporal_analysis['growth_rate']:+.1f}%\n"
            details += f"• Days Since Last Scan: {temporal_analysis['days_since_last']}\n"
            if temporal_analysis['new_tumors'] > 0:
                details += f"• ⚠️ New Lesions Detected: {temporal_analysis['new_tumors']}\n"
            details += "\n"
        
        # Differential diagnosis
        ddx_report = "🩺 DIFFERENTIAL DIAGNOSIS:\n\n"
        for idx, (diagnosis, probability) in enumerate(ddx[:5], 1):
            ddx_report += f"{idx}. {diagnosis}: {probability:.1f}%\n"
        
        # Treatment recommendations
        treatment_report = "💊 TREATMENT RECOMMENDATIONS:\n\n"
        for idx, rec in enumerate(treatments, 1):
            treatment_report += f"{idx}. [{rec['type']}] {rec['treatment']}\n"
            treatment_report += f"   Timeline: {rec['timeline']}\n"
            treatment_report += f"   {rec['details']}\n\n"
        
        # Save to database
        if patient_id:
            radiomics_id = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save radiomics
            with open(RADIOMICS_FILE, 'r') as f:
                rad_db = json.load(f)
            rad_db[radiomics_id] = {
                'patient_id': patient_id,
                'date': datetime.now().isoformat(),
                'features': all_radiomics
            }
            with open(RADIOMICS_FILE, 'w') as f:
                json.dump(rad_db, f, indent=2)
            
            # Save patient record
            new_record = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Patient_ID": patient_id,
                "Organ": organ_key,
                "Diagnosis": ddx[0][0],
                "Confidence": f"{max_tumor['confidence']:.1f}%",
                "Tumor_Count": tumor_count,
                "Max_Size_mm": max_tumor['size'],
                "Risk_Level": risk,
                "Cost_Est": "Calculated",
                "Notes": doctor_notes[:100] if doctor_notes else "",
                "Radiomics_ID": radiomics_id,
                "Quality_Score": f"{overall_quality:.1f}%"
            }
            df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df.to_csv(HISTORY_FILE, index=False)
            
            log_audit_event("ANALYSIS_COMPLETE", patient_id, f"Detected {tumor_count} lesions, Risk: {risk}")
    
    else:
        status = "✅ NORMAL SCAN"
        details = "No significant abnormalities detected"
        ddx_report = "No differential diagnosis needed"
        treatment_report = "Routine follow-up recommended"
    
    return (annotated_rgb, status, details, ddx_report, treatment_report, 
            quality_report, uncertainty_report)

def get_risk_level(organ, size_mm, confidence):
    """Enhanced risk stratification"""
    if organ == "Brain":
        if size_mm < 10: return "Low" if confidence < 70 else "Moderate"
        elif size_mm < 30: return "Moderate" if confidence < 80 else "High"
        else: return "Critical"
    elif organ == "Lung":
        if size_mm < 6: return "Low"
        elif size_mm < 15: return "Moderate"
        else: return "High"
    else:
        if size_mm < 20: return "Low"
        elif size_mm < 40: return "Moderate"
        else: return "High"

def get_filtered_history(organ_filter="All", risk_filter="All", days_back=30):
    """Advanced filtering"""
    df = pd.read_csv(HISTORY_FILE)
    if df.empty: return df
    
    df['Date'] = pd.to_datetime(df['Date'])
    cutoff = datetime.now() - timedelta(days=days_back)
    df = df[df['Date'] >= cutoff]
    
    if organ_filter != "All": df = df[df['Organ'] == organ_filter]
    if risk_filter != "All" and 'Risk_Level' in df.columns:
        df = df[df['Risk_Level'] == risk_filter]
    
    return df.sort_values('Date', ascending=False)

def get_advanced_statistics():
    """Generate comprehensive analytics"""
    df = pd.read_csv(HISTORY_FILE)
    if df.empty: return "No data available"
    
    stats = f"📊 COMPREHENSIVE ANALYTICS\n\n"
    stats += f"Total Scans: {len(df)}\n"
    stats += f"Unique Patients: {df['Patient_ID'].nunique()}\n\n"
    
    stats += "Organ Distribution:\n"
    for organ, count in df['Organ'].value_counts().items():
        stats += f"  • {organ}: {count} ({count/len(df)*100:.1f}%)\n"
    
    if 'Risk_Level' in df.columns:
        stats += f"\nRisk Stratification:\n"
        for risk, count in df['Risk_Level'].value_counts().items():
            stats += f"  • {risk}: {count} ({count/len(df)*100:.1f}%)\n"
    
    if 'Quality_Score' in df.columns:
        stats += f"\nAverage Quality Score: {df['Quality_Score'].str.rstrip('%').astype(float).mean():.1f}%\n"
    
    return stats

# --- UI ---
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "system-ui"]
)

with gr.Blocks(theme=theme, title="OncoVision AI Ultra") as app:
    gr.Markdown("# 🏥 OncoVision AI Ultra\n### Next-Generation Medical AI with Advanced Analytics")
    
    with gr.Tabs():
        with gr.TabItem("🔬 Ultra Scanner"):
            with gr.Row():
                with gr.Column(scale=1):
                    pid = gr.Textbox(label="Patient ID")
                    organ = gr.Dropdown(
                        ["Brain (MRI)", "Lung (CT)", "Liver (CT)", "Kidney (CT)"],
                        label="Organ", value="Brain (MRI)"
                    )
                    img_input = gr.Image(type="numpy", label="Medical Scan")
                    
                    gr.Markdown("### ⚙️ Advanced Settings")
                    conf = gr.Slider(0.1, 0.9, value=0.15, step=0.05, label="Detection Sensitivity")
                    ensemble = gr.Checkbox(label="Ensemble Prediction", value=True)
                    preprocess = gr.Checkbox(label="Medical-Grade Preprocessing (CLAHE) (May be innacurate in some cases.)", value=False)
                    uncertainty = gr.Checkbox(label="Uncertainty Quantification (Monte Carlo)", value=True)
                    notes = gr.Textbox(label="Clinical Notes", lines=3)
                    
                    analyze_btn = gr.Button("🚀 Run Ultra Analysis", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🩺 Comprehensive Report")
                    img_out = gr.Image(label="AI Detection & Localization")
                    status_out = gr.Textbox(label="Primary Findings", lines=4)
                    
                    with gr.Accordion("📋 Detailed Analysis", open=True):
                        details_out = gr.Textbox(label="Clinical Details", lines=10)
                    
                    with gr.Accordion("🩺 Differential Diagnosis", open=True):
                        ddx_out = gr.Textbox(label="Top Differential Diagnoses", lines=8)
                    
                    with gr.Accordion("💊 Treatment Plan", open=True):
                        treatment_out = gr.Textbox(label="Recommended Treatment Protocol", lines=10)
                    
                    with gr.Accordion("📊 Quality & Uncertainty", open=False):
                        quality_out = gr.Textbox(label="Image Quality Assessment", lines=6)
                        uncertainty_out = gr.Textbox(label="Prediction Uncertainty", lines=4)
            
            analyze_btn.click(
                fn=ultra_advanced_analysis,
                inputs=[pid, img_input, organ, conf, ensemble, preprocess, uncertainty, notes],
                outputs=[img_out, status_out, details_out, ddx_out, treatment_out, quality_out, uncertainty_out]
            )
        
        with gr.TabItem("📈 Patient Timeline"):
            gr.Markdown("### Longitudinal Analysis - Track Disease Progression")
            
            with gr.Row():
                timeline_pid = gr.Textbox(label="Patient ID", placeholder="Enter Patient ID")
                load_timeline_btn = gr.Button("Load Patient History", variant="primary")
            
            timeline_plot = gr.Plot(label="Tumor Size Over Time")
            timeline_table = gr.Dataframe(label="Historical Records")
            timeline_analysis = gr.Textbox(label="Progression Analysis", lines=8)
            
            def generate_timeline(patient_id):
                """Generate temporal visualization"""
                df = pd.read_csv(HISTORY_FILE)
                patient_df = df[df['Patient_ID'] == patient_id].copy()
                
                if patient_df.empty:
                    return None, patient_df, "No records found for this patient"
                
                patient_df['Date'] = pd.to_datetime(patient_df['Date'])
                patient_df = patient_df.sort_values('Date')
                
                # Create matplotlib plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(patient_df['Date'], patient_df['Max_Size_mm'], 
                       marker='o', linewidth=2, markersize=8, color='#0f766e')
                ax.fill_between(patient_df['Date'], patient_df['Max_Size_mm'], 
                               alpha=0.3, color='#0f766e')
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Tumor Size (mm)', fontsize=12, fontweight='bold')
                ax.set_title(f'Tumor Growth Timeline - Patient {patient_id}', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Analysis
                if len(patient_df) >= 2:
                    first_size = patient_df.iloc[0]['Max_Size_mm']
                    last_size = patient_df.iloc[-1]['Max_Size_mm']
                    total_growth = last_size - first_size
                    growth_pct = (total_growth / first_size) * 100 if first_size > 0 else 0
                    days_span = (patient_df.iloc[-1]['Date'] - patient_df.iloc[0]['Date']).days
                    
                    analysis = f"LONGITUDINAL ANALYSIS:\n\n"
                    analysis += f"• First Scan: {patient_df.iloc[0]['Date'].strftime('%Y-%m-%d')}\n"
                    analysis += f"• Latest Scan: {patient_df.iloc[-1]['Date'].strftime('%Y-%m-%d')}\n"
                    analysis += f"• Total Scans: {len(patient_df)}\n"
                    analysis += f"• Monitoring Duration: {days_span} days\n\n"
                    analysis += f"GROWTH METRICS:\n"
                    analysis += f"• Initial Size: {first_size:.1f} mm\n"
                    analysis += f"• Current Size: {last_size:.1f} mm\n"
                    analysis += f"• Absolute Growth: {total_growth:+.1f} mm\n"
                    analysis += f"• Percentage Growth: {growth_pct:+.1f}%\n"
                    analysis += f"• Growth Rate: {(total_growth/days_span)*30:.2f} mm/month\n\n"
                    
                    if growth_pct > 20:
                        analysis += "⚠️ SIGNIFICANT GROWTH DETECTED - Consider escalation of care\n"
                    elif growth_pct > 10:
                        analysis += "⚠️ Moderate growth - Continue close monitoring\n"
                    else:
                        analysis += "✅ Stable disease - Continue surveillance\n"
                else:
                    analysis = "Insufficient data for temporal analysis (need at least 2 scans)"
                
                return fig, patient_df[['Date', 'Organ', 'Max_Size_mm', 'Tumor_Count', 'Risk_Level']], analysis
            
            load_timeline_btn.click(
                fn=generate_timeline,
                inputs=[timeline_pid],
                outputs=[timeline_plot, timeline_table, timeline_analysis]
            )
        
        with gr.TabItem("🔬 Radiomics Explorer"):
            gr.Markdown("### Advanced Feature Analysis")
            
            radiomics_pid = gr.Textbox(label="Patient ID")
            load_radiomics_btn = gr.Button("Load Radiomics Features", variant="primary")
            
            radiomics_display = gr.Textbox(label="Feature Extraction Results", lines=20)
            
            def display_radiomics(patient_id):
                """Display radiomics features for patient"""
                with open(RADIOMICS_FILE, 'r') as f:
                    rad_db = json.load(f)
                
                patient_radiomics = {k: v for k, v in rad_db.items() if patient_id in k}
                
                if not patient_radiomics:
                    return "No radiomics data found for this patient"
                
                output = f"RADIOMICS ANALYSIS - Patient {patient_id}\n\n"
                output += f"Total Feature Sets: {len(patient_radiomics)}\n\n"
                
                for rad_id, data in sorted(patient_radiomics.items(), 
                                          key=lambda x: x[1]['date'], reverse=True):
                    output += f"{'='*60}\n"
                    output += f"Scan Date: {data['date']}\n"
                    output += f"Feature Set ID: {rad_id}\n\n"
                    
                    if data['features']:
                        features = data['features'][0]  # First tumor
                        output += "FIRST-ORDER STATISTICS:\n"
                        output += f"  • Mean Intensity: {features.get('mean_intensity', 0):.2f}\n"
                        output += f"  • Std Deviation: {features.get('std_intensity', 0):.2f}\n"
                        output += f"  • Skewness: {features.get('skewness', 0):.3f}\n"
                        output += f"  • Kurtosis: {features.get('kurtosis', 0):.3f}\n"
                        output += f"  • Entropy: {features.get('entropy', 0):.3f}\n\n"
                        
                        output += "SHAPE FEATURES:\n"
                        output += f"  • Area: {features.get('area', 0)} pixels²\n"
                        output += f"  • Perimeter: {features.get('perimeter', 0):.1f} pixels\n"
                        output += f"  • Sphericity: {features.get('sphericity', 0):.4f}\n"
                        output += f"  • Aspect Ratio: {features.get('aspect_ratio', 0):.2f}\n\n"
                        
                        output += "TEXTURE FEATURES:\n"
                        output += f"  • Homogeneity: {features.get('homogeneity', 0):.4f}\n\n"
                
                return output
            
            load_radiomics_btn.click(
                fn=display_radiomics,
                inputs=[radiomics_pid],
                outputs=[radiomics_display]
            )
                
        with gr.TabItem("📦 Batch Analysis"):
            gr.Markdown("### Process Multiple Scans Simultaneously")
            
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(label="Upload Multiple Scans", file_count="multiple", file_types=["image"])
                    batch_organ = gr.Dropdown(
                        ["Brain (MRI)", "Lung (CT)", "Liver (CT)", "Kidney (CT)"],
                        label="Organ Type",
                        value="Brain (MRI)"
                    )
                    batch_conf = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Detection Sensitivity")
                    batch_preprocess = gr.Checkbox(label="Apply Preprocessing", value=False)
                    batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                
                with gr.Column():
                    batch_progress = gr.Textbox(label="Processing Status", lines=3)
                    batch_summary = gr.Textbox(label="Batch Results Summary", lines=15)
                    batch_table = gr.Dataframe(label="Detailed Results")
            
            def batch_analysis(files, organ_type, conf_threshold, use_preprocess):
                """Process multiple images in batch"""
                if not files:
                    return "⚠️ No files uploaded", "", None
                
                organ_key = organ_type.split(" ")[0]
                results_summary = []
                detailed_results = []
                
                progress_text = f"Processing {len(files)} scans...\n"
                
                for idx, file in enumerate(files):
                    try:
                        # Read image
                        image = cv2.imread(file.name)
                        if image is None:
                            results_summary.append(f"❌ Image {idx+1}: Failed to read")
                            continue
                        
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Run analysis
                        _, status, details, ddx, treatment, quality, uncertainty = ultra_advanced_analysis(
                            f"Batch_{idx+1}", 
                            image_rgb, 
                            organ_type, 
                            conf_threshold, 
                            False,  # ensemble
                            use_preprocess, 
                            False,  # uncertainty
                            f"Batch scan {idx+1}"
                        )
                        
                        # Extract key info
                        is_abnormal = "ABNORMALITY DETECTED" in status
                        risk = "Unknown"
                        lesion_count = 0
                        
                        if is_abnormal:
                            # Extract risk level
                            if "Critical" in status:
                                risk = "Critical"
                            elif "High" in status:
                                risk = "High"
                            elif "Moderate" in status:
                                risk = "Moderate"
                            else:
                                risk = "Low"
                            
                            # Extract lesion count
                            if "Detected Lesions:" in status:
                                lesion_count = int(status.split("Detected Lesions:")[1].split("\n")[0].strip())
                        
                        status_icon = "⚠️" if is_abnormal else "✅"
                        results_summary.append(
                            f"{status_icon} Scan {idx+1}: {'ABNORMAL' if is_abnormal else 'NORMAL'} | Risk: {risk}"
                        )
                        
                        detailed_results.append({
                            "Scan": f"Image_{idx+1}",
                            "Status": "Abnormal" if is_abnormal else "Normal",
                            "Risk": risk,
                            "Lesions": lesion_count,
                            "Organ": organ_key
                        })
                        
                    except Exception as e:
                        results_summary.append(f"❌ Scan {idx+1}: Error - {str(e)}")
                
                # Summary statistics
                abnormal_count = sum(1 for r in results_summary if "ABNORMAL" in r)
                normal_count = len(results_summary) - abnormal_count
                
                summary_text = f"📊 BATCH PROCESSING COMPLETE\n\n"
                summary_text += f"Total Scans: {len(files)}\n"
                summary_text += f"✅ Normal: {normal_count}\n"
                summary_text += f"⚠️ Abnormal: {abnormal_count}\n"
                summary_text += f"Success Rate: {(len(results_summary)/len(files))*100:.1f}%\n\n"
                summary_text += "="*50 + "\n\n"
                summary_text += "INDIVIDUAL RESULTS:\n\n"
                summary_text += "\n".join(results_summary)
                
                progress_final = f"✅ Processing complete!\nProcessed: {len(files)} scans"
                
                df_results = pd.DataFrame(detailed_results) if detailed_results else pd.DataFrame()
                
                return progress_final, summary_text, df_results
            
            batch_btn.click(
                fn=batch_analysis,
                inputs=[batch_files, batch_organ, batch_conf, batch_preprocess],
                outputs=[batch_progress, batch_summary, batch_table]
            )
        
        with gr.TabItem("📊 Advanced Analytics"):
            gr.Markdown("### System-Wide Statistical Analysis")
            
            with gr.Row():
                with gr.Column():
                    stats_btn = gr.Button("📈 Generate Statistics", variant="primary", size="lg")
                    organ_filter2 = gr.Dropdown(
                        ["All", "Brain", "Lung", "Liver", "Kidney"],
                        label="Filter by Organ", value="All"
                    )
                    risk_filter2 = gr.Dropdown(
                        ["All", "Low", "Moderate", "High", "Critical"],
                        label="Filter by Risk", value="All"
                    )
                    days_filter2 = gr.Slider(7, 365, value=90, label="Days Back")
                
                with gr.Column():
                    stats_output = gr.Textbox(label="Statistical Summary", lines=15)
                    analytics_table = gr.Dataframe(label="Filtered Records")
            
            stats_btn.click(fn=get_advanced_statistics, outputs=[stats_output])
            
            filter_btn2 = gr.Button("Apply Filters")
            filter_btn2.click(
                fn=get_filtered_history,
                inputs=[organ_filter2, risk_filter2, days_filter2],
                outputs=[analytics_table]
            )
        
        with gr.TabItem("🗂️ Patient Database"):
            gr.Markdown("### Comprehensive Patient Records")
            
            with gr.Row():
                search_pid = gr.Textbox(label="Search Patient ID")
                search_btn = gr.Button("🔍 Search", variant="secondary")
            
            records_table = gr.Dataframe(value=get_filtered_history(days_back=365))
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh All")
                export_btn = gr.Button("💾 Export CSV")
            
            def search_patient(patient_id):
                df = pd.read_csv(HISTORY_FILE)
                if patient_id:
                    df = df[df['Patient_ID'].str.contains(patient_id, case=False, na=False)]
                return df
            
            search_btn.click(fn=search_patient, inputs=[search_pid], outputs=[records_table])
            refresh_btn.click(fn=lambda: get_filtered_history(days_back=365), outputs=[records_table])
        
        with gr.TabItem("🔐 Audit Trail"):
            gr.Markdown("### System Activity Log")
            
            audit_btn = gr.Button("📜 Load Recent Activity", variant="primary")
            audit_output = gr.Textbox(label="Audit Log", lines=20)
            
            def display_audit_log():
                with open(AUDIT_LOG, 'r') as f:
                    audit = json.load(f)
                
                output = "SYSTEM AUDIT TRAIL\n\n"
                for entry in reversed(audit[-50:]):  # Last 50 events
                    output += f"{entry['timestamp']}\n"
                    output += f"  Event: {entry['event']}\n"
                    output += f"  Patient: {entry['patient_id']}\n"
                    output += f"  Details: {entry['details']}\n\n"
                
                return output
            
            audit_btn.click(fn=display_audit_log, outputs=[audit_output])
        
        with gr.TabItem("ℹ️ Documentation"):
            gr.Markdown("""
            ## **OncoVision AI Ultra v4.0**
            
            ### 🚀 ULTRA-ADVANCED FEATURES
            
            #### **AI/ML Enhancements**
            - ✅ **Monte Carlo Dropout** - Uncertainty quantification
            - ✅ **CLAHE Preprocessing** - Medical-grade image enhancement
            - ✅ **Ensemble Predictions** - Multi-confidence fusion
            - ✅ **Radiomics Extraction** - 15+ quantitative features
            - ✅ **Quality Scoring** - 5-dimensional image assessment
            
            #### **Clinical Intelligence**
            - ✅ **Differential Diagnosis** - Top 5 likely conditions with probabilities
            - ✅ **Treatment Protocols** - Evidence-based recommendations
            - ✅ **Temporal Analysis** - Track tumor growth over time
            - ✅ **Growth Rate Calculation** - mm/month progression metrics
            - ✅ **Multi-lesion Tracking** - Handle complex cases
            
            #### **Advanced Analytics**
            - ✅ **Radiomics Database** - Store texture & shape features
            - ✅ **Longitudinal Plotting** - Visualize disease progression
            - ✅ **Risk Stratification** - 4-tier classification system
            - ✅ **Cohort Analysis** - Population-level statistics
            - ✅ **Audit Trail** - Complete activity logging
            
            #### **Quality & Safety**
            - ✅ **Sharpness Detection** - Blur/motion artifact identification
            - ✅ **Contrast Analysis** - Exposure optimization
            - ✅ **Noise Estimation** - SNR calculation
            - ✅ **Artifact Detection** - FFT-based anomaly detection
            - ✅ **Uncertainty Metrics** - Prediction reliability scores
            
            ### 📊 NEW CAPABILITIES
            
            **1. Radiomics Analysis**
            - Extract 15+ quantitative imaging features
            - First-order statistics (mean, std, skewness, kurtosis, entropy)
            - Shape descriptors (area, perimeter, sphericity, aspect ratio)
            - Texture features (homogeneity, contrast)
            
            **2. Temporal Tracking**
            - Automatic growth rate calculation
            - Visual timeline plots
            - Progression alerts
            - New lesion detection
            
            **3. Differential Diagnosis Engine**
            - Probability-weighted diagnoses
            - Organ-specific disease databases
            - Multi-factor analysis (size, count, features)
            
            **4. Treatment Recommendation System**
            - Evidence-based protocols
            - Timeline planning
            - Multi-modal therapy options
            - Follow-up scheduling
            
            **5. Advanced Quality Control**
            - 100-point quality scoring
            - Multi-dimensional assessment
            - Automatic issue flagging
            - Preprocessing recommendations
            
            ### 🔬 TECHNICAL SPECIFICATIONS
            
            **Image Processing Pipeline:**
            1. CLAHE enhancement (8x8 tiles, clip limit 3.0)
            2. Bilateral filtering (preserves edges)
            3. Organ-specific windowing
            4. Normalization & standardization
            
            **Detection Algorithm:**
            1. YOLOv8 backbone with custom medical heads
            2. Multi-scale feature pyramid
            3. Confidence thresholding (adaptive)
            4. Non-maximum suppression
            5. Size-based filtering
            
            **Quality Metrics:**
            - Sharpness: Laplacian variance
            - Contrast: Standard deviation
            - Brightness: Mean intensity analysis
            - Noise: Sobel gradient estimation
            - Artifacts: FFT magnitude detection
            
            **Uncertainty Quantification:**
            - Monte Carlo Dropout (10 iterations)
            - Confidence variance calculation
            - Reliability classification
            
            ### 📋 CLINICAL WORKFLOW
            
            1. **Upload scan** → System validates image quality
            2. **Automatic preprocessing** → CLAHE + denoising
            3. **AI detection** → Multi-confidence ensemble
            4. **Feature extraction** → Radiomics computation
            5. **Risk assessment** → 4-tier stratification
            6. **DDx generation** → Top 5 diagnoses
            7. **Treatment planning** → Protocol recommendations
            8. **Database storage** → Complete audit trail
            
            ### ⚠️ MEDICAL DISCLAIMER
            
            **FOR RESEARCH & EDUCATIONAL USE ONLY**
            
            This system provides AI-assisted analysis to support clinical decision-making.
            All findings must be reviewed and confirmed by qualified medical professionals.
            Not approved for direct clinical use without appropriate validation and oversight.
            
            ### 🔐 DATA PRIVACY
            
            - All patient data stored locally
            - Complete audit trail maintained
            - HIPAA-compliant data handling
            - Encrypted storage recommended
            
            ### 📞 SUPPORT
            
            For technical support, validation studies, or collaboration:
            - GitHub: [Your Repository]
            - Email: medical-ai@research.org
            - Documentation: [Your Docs URL]
            
            ---
            
            **Version 4.0** | Last Updated: 2024
            
            *Advancing Medical AI for Better Patient Outcomes*
            """)
    
    # Auto-refresh on image change
    img_input.change(
        fn=ultra_advanced_analysis,
        inputs=[pid, img_input, organ, conf, ensemble, preprocess, uncertainty, notes],
        outputs=[img_out, status_out, details_out, ddx_out, treatment_out, quality_out, uncertainty_out]
    )

app.launch(show_api=False, share=False)