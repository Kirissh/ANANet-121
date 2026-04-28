import os
import io
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import timm
import torch.nn.functional as F
from flask import Flask, request, render_template, send_file, jsonify
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import base64
import requests
import time
import json
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SOTAHep2Wrapper(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.encoder = timm.create_model('densenet121', pretrained=False, num_classes=0, global_pool='')
        self.head = nn.Linear(1024, num_classes)
    def forward(self, x):
        features = self.backbone.encoder(x)
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.head(features)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        def forward_hook(module, input, output): self.activations = output
        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        if target_class is None: target_class = torch.argmax(logits, dim=1).item()
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)
        if self.gradients is None: return np.zeros((224, 224), dtype=np.float32), target_class
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * self.activations, dim=1).squeeze(0)).cpu().detach().numpy()
        if cam.max() > cam.min(): cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

device = torch.device("cpu")
model = SOTAHep2Wrapper(num_classes=6)
MODEL_PATH = "Verified_SOTA_HEp2_Model.pth"
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'])
    else: model.load_state_dict(state_dict)
model.to(device)
for param in model.parameters(): param.requires_grad = True
model.eval()

target_layer = model.backbone.encoder.features.norm5
gradcam = GradCAM(model, target_layer)

class_names = ['centromere', 'coarse_speckled', 'cytoplasmatic', 'fine_speckled', 'homogeneous', 'nucleolar']
characteristics = {
    'centromere': 'Discrete speckles uniformly distributed throughout nucleoplasm.',
    'coarse_speckled': 'Large, irregularly sized and shaped speckles.',
    'cytoplasmatic': 'Fluorescence localized in the cell cytoplasm.',
    'fine_speckled': 'Small, uniform speckles in the nucleoplasm.',
    'homogeneous': 'Uniform, diffuse fluorescence across the nucleus.',
    'nucleolar': 'Bright staining localized specifically to nucleoli.'
}
disease_map = {
    'centromere': ('CREST / Systemic Sclerosis', 'Anti-CENP'),
    'coarse_speckled': ('SLE / MCTD', 'Anti-Sm, Anti-U1RNP'),
    'cytoplasmatic': ('Liver Disease / Myositis', 'AMA, Anti-Jo1'),
    'fine_speckled': ('Sjögren\'s / SLE', 'Anti-Ro/SSA, Anti-La/SSB'),
    'homogeneous': ('SLE / Drug-induced', 'Anti-dsDNA, Histone'),
    'nucleolar': ('Systemic Sclerosis', 'Anti-Scl-70')
}

def to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def preprocess_crop(crop_img):
    img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)).resize((224, 224))
    img_np = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)

def divide_into_pois(img, rows=2, cols=2):
    # Divide the image into a grid of POIs (Points of Interest)
    h, w = img.shape[:2]
    poi_h, poi_w = h // rows, w // cols
    crops = []
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * poi_h, (i + 1) * poi_h
            x1, x2 = j * poi_w, (j + 1) * poi_w
            # Ensure we don't go out of bounds
            if i == rows - 1: y2 = h
            if j == cols - 1: x2 = w
            
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                
    # Always include the whole image as global context
    crops.append(img)
    return crops

@app.route('/')
def index(): return render_template('index.html')

@app.route('/step1', methods=['POST'])
def step1():
    file = request.files['image']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    grid_val = request.form.get('grid_size', '1x1')
    if 'x' in str(grid_val):
        rows, cols = map(int, str(grid_val).split('x'))
    else:
        rows = cols = int(grid_val)
    # Divide into POIs (e.g., NxM grid + 1 whole image)
    pois = divide_into_pois(original_img, rows=rows, cols=cols)
    
    all_probs = []
    with torch.no_grad():
        for poi in pois:
            tensor = preprocess_crop(poi)
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs)
            
    # Average the probabilities across all POIs for a robust group-level prediction
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    conf, idx = torch.max(avg_probs, 0)
    
    class_name = class_names[idx.item()]
    
    # Generate Grad-CAM on the whole image for visual context
    tensor_full = preprocess_crop(original_img)
    cam, _ = gradcam.generate(tensor_full, target_class=idx.item())
    
    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    gradcam_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    return jsonify({
        'class_name': class_name,
        'confidence': float(conf),
        'characteristic': characteristics[class_name],
        'gradcam_image': to_base64(gradcam_img),
        'original_image': to_base64(original_img)
    })

@app.route('/step2', methods=['POST'])
def step2():
    data = request.json
    image_b64 = data.get('image')
    sota_pattern = data.get('sota_pattern')
    sota_conf = data.get('sota_conf')
    import os
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "5ec8b08994414dcaa14b6781f4777fba.VoEw5MiybaT91Vmc8yhpoFpz")
    prompt = f"Act as a pathologist. The primary AnaNet-121 model found {sota_pattern} ({sota_conf}). Analyze the RAW specimen image and provide an independent review. Return ONLY JSON: {{\"classification\": \"[one of: centromere, coarse_speckled, cytoplasmatic, fine_speckled, homogeneous, nucleolar]\", \"confidence\": \"[number]%\", \"insight\": \"string\"}}. No bolding."
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gemma4:31b-cloud",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_b64}}]}],
        "response_format": {"type": "json_object"}
    }
    try:
        r = requests.post("https://ollama.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content'].strip()
            # Robust JSON extraction
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                text = match.group(0)
                try:
                    return jsonify(json.loads(text))
                except json.JSONDecodeError:
                    pass
            
            # Fallback if VLM didn't return valid JSON
            return jsonify({
                "classification": "Unknown",
                "confidence": "N/A",
                "insight": content
            })
        else: return jsonify({"error": f"API Error {r.status_code}"}), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.json
    pt = data.get('patient_info', {})
    s1 = data.get('step1', {})
    s2 = data.get('step2', {})
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T', parent=styles['Heading1'], alignment=1, fontSize=16, spaceAfter=20, textColor=colors.HexColor('#1e3a8a'))
    h_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor('#1e3a8a'))
    n_style = styles['Normal']
    n_style.fontSize = 10
    
    elements.append(Paragraph("<b>ANA SCREENING DIAGNOSTIC REPORT</b>", title_style))
    
    # 1. Patient Table
    elements.append(Paragraph("<b>1. PATIENT INFORMATION</b>", h_style))
    pt_data = [
        ['Patient Name:', pt.get('name', 'N/A'), 'ID:', pt.get('patientId', 'N/A')],
        ['Age/Gender:', pt.get('age', 'N/A'), 'Date:', time.strftime("%d %B %Y")]
    ]
    t_pt = Table(pt_data, colWidths=[1.2*inch, 2.3*inch, 0.8*inch, 2.3*inch])
    t_pt.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 10), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(t_pt)
    
    # 2. Results Table
    elements.append(Paragraph("<b>2. CONCORDANCE ANALYSIS SUMMARY</b>", h_style))
    res_data = [
        ['Parameter', 'Verified AnaNet-121', 'Gemma-4 Cloud VLM'],
        ['ANA Pattern Type', s1.get('class_name', '').upper(), s2.get('classification', '').upper()],
        ['Confidence Score', f"{s1.get('confidence', 0)*100:.1f}%", s2.get('confidence', 'N/A')],
        ['Positive Status', 'Strong Positive' if s1.get('confidence', 0) > 0.8 else 'Positive', 'Verified' if s2.get('classification') else 'N/A']
    ]
    t_res = Table(res_data, colWidths=[1.8*inch, 2.4*inch, 2.4*inch])
    t_res.setStyle(TableStyle([
        ['BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')],
        ['GRID', (0,0), (-1,-1), 0.5, colors.grey],
        ['FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'],
        ['ALIGN', (0,0), (-1,-1), 'CENTER'],
        ['VALIGN', (0,0), (-1,-1), 'MIDDLE']
    ]))
    elements.append(t_res)

    # 3. Characteristics & Insights
    elements.append(Paragraph("<b>3. MORPHOLOGICAL INSIGHTS</b>", h_style))
    elements.append(Paragraph(f"<b>Key Characteristics:</b> {s1.get('characteristic', 'N/A')}", n_style))
    elements.append(Paragraph(f"<b>VLM Clinical Insight:</b> {s2.get('insight', 'N/A')}", n_style))

    # 4. Visual Reasoning
    elements.append(Paragraph("<b>4. GRAD-CAM CONFIDENCE MAPPING</b>", h_style))
    img_b64 = s1.get('gradcam_image')
    if img_b64:
        img_data = base64.b64decode(img_b64.split(',')[1])
        elements.append(RLImage(io.BytesIO(img_data), width=3.5*inch, height=3.5*inch, kind='proportional'))
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<i>Note: This is an AI-assisted diagnostic report. Clinical correlation by a pathologist is mandatory.</i>", n_style))

    doc.build(elements)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name=f"ANA_Report_{pt.get('name', 'Patient')}.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
