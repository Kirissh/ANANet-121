# ANA Diagnostic Pipeline: Technical Deep Dive (AnaNet-121)

This document provides a comprehensive technical overview of the AI-powered ANA (Antinuclear Antibody) screening pipeline, covering the model architecture, training strategy, and application logic.

---

## 1. Model Architecture: AnaNet-121
The core of the diagnostic system is **AnaNet-121**, a deep convolutional neural network based on the **DenseNet-121** architecture.

### **Why DenseNet-121?**
*   **Feature Reuse**: Unlike ResNet, where features are added, DenseNet concatenates features. Each layer has direct access to the gradients and original input information from all preceding layers.
*   **Textural Precision**: HEp-2 patterns (e.g., "Fine Speckled" vs. "Coarse Speckled") differ by minute textural granules. Dense connectivity ensures that low-level textural features are preserved throughout the network depth.
*   **Efficiency**: With only ~7M parameters, it outperforms much larger models like ViT-Base (86M params) on specialized medical datasets like MIVIA.

### **Modifications for ANA Use-Case**
*   **Custom Head**: The standard ImageNet head was replaced with a specialized **Linear Classification Head** ($1024 \to 6$ classes).
*   **Global Average Pooling (GAP)**: Replaces fully connected layers to reduce parameter count and minimize overfitting on small microscopic datasets.

---

## 2. Training Strategy and Preprocessing
The model was trained on the **MIVIA HEp-2 Dataset** (1,455 images) using a rigorous scientific protocol.

### **Image Preprocessing**
*   **Spatial Normalization**: All images are resized to $224 \times 224$ pixels.
*   **Pixel Calibration**: Normalized using ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) to leverage transfer learning features.
*   **Aspect Focus**: The model focuses on **Intensity Distribution** (brightness of patterns) and **Spatial Frequency** (the "grain" or texture of the speckles).

### **Data Augmentation (Online)**
To ensure the model is invariant to slide orientation, we applied:
*   **Random 180° Rotations**: HEp-2 cells are biologically isotropic.
*   **Horizontal/Vertical Flips**: Maximizes the variety of morphological orientations.
*   **Elastic Transformations**: Simulates slight variations in cell shape during staining/fixation.

### **Staged Training Protocol**
1.  **Phase 1: Feature Saturation**: Full fine-tuning of the backbone to reach a peak F1-score of **99.76%**.
2.  **Phase 2: Weight Recovery & Freezing**: 
    *   **Freezing Backbone**: The convolutional backbone is "locked" to prevent "catastrophic forgetting" of the high-quality features learned in Phase 1.
    *   **Weights Recovery**: We restore the best-performing weights from the saturation phase and re-calibrate only the classification head to ensure production stability.

---

## 3. Explainability with Grad-CAM
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to generate the "Clinical Audit Trail."
*   **Mechanism**: It uses the gradients of the target class (e.g., "Nucleolar") flowing into the final convolutional layer (`norm5`) to produce a localization map.
*   **Visual Datapoints**: The pipeline extracts local maxima from these heatmaps to place "datapoints" over the image, showing the pathologist exactly where the AI "looked" to make its decision.

---

## 4. Application Logic (`app.py`) Flow
The application implements a high-precision sequential diagnostic journey:

### **Step 1: Morphological Extraction (AnaNet-121)**
*   **POI Grid Division**: The raw input image is mathematically divided into a grid (1x1, 2x2, or 3x6).
*   **Consensus Prediction**: Instead of one check, the model runs inference on **every POI patch**. The probability distributions are averaged (`torch.mean`) to reach a robust final diagnosis that covers both single cells and large clusters.
*   **Intervention Gate**: If the consensus confidence is **< 70%**, the system halts for a manual Pathologist Verification.

### **Step 2: VLM Concordance (Gemma-4)**
*   **Independent Review**: The **raw original image** is sent to the Gemma-4 Vision-Language Model.
*   **Cross-Verification**: Gemma-4 provides an independent classification and a "Comparative Insight," weighing its broad-view analysis against AnaNet's texture-focused analysis.

### **Step 3: Structured Reporting**
*   **ReportLab Engine**: Generates a professional PDF containing a **Concordance Table**.
*   **Data Capture**: Includes AnaNet pattern/confidence, Gemma pattern/confidence, Clinical Insights, and Grad-CAM visual evidence.

---

---

## 5. How to Run the Pipeline

### **Prerequisites**
1. **Python 3.10+** installed.
2. **Ollama** installed (for local VLM access) or an **Ollama API Key** (for cloud access).
3. **Verified_SOTA_HEp2_Model.pth** must be present in the root directory.

### **Installation**
Open your terminal in the project directory and run:
```bash
pip install torch torchvision timm opencv-python flask requests reportlab numpy Pillow
```

### **Running the Application**
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

### **Usage Workflow**
1. **Upload**: Drag and drop a raw HEp-2 specimen image.
2. **Configure**: Select your preferred **POI Division** (e.g., 3x6 Grid for complex cell clusters).
3. **Analyze**: Click "Initialize AnaNet-121 Pipeline."
4. **Verify**: If confidence is low, perform the manual Pathologist Verification.
5. **VLM Consensus**: Proceed to Gemma-4 for the second opinion.
6. **Report**: Download the final structured PDF for clinical records.

---
