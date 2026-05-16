# 🏥 Pneumonia Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An AI-powered system for detecting pneumonia from chest X-ray images using deep learning. The system leverages the VGG16 architecture pre-trained on ImageNet to classify X-rays into three categories: **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia**.

## Features

-  **Deep Learning Model**: VGG16-based architecture with custom classification layers
-  **High Accuracy**: Achieves excellent performance on pneumonia detection
-  **Web Interface**: User-friendly Streamlit application for real-time predictions
-  **Data Augmentation**: Comprehensive image preprocessing and augmentation
-  **Model Evaluation**: Detailed metrics including precision, recall, F1-score, and confusion matrix
-  **Professional Code**: Well-structured, documented, and production-ready

##  Model Architecture

```
Input (224x224x3)
    ↓
VGG16 Base Model (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense Layer (512 units) + Dropout
    ↓
Dense Layer (256 units) + Dropout
    ↓
Output Layer (3 classes - Softmax)
```

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vigneshreddy99/Pneumonia-Detection-Using-Machine-Learning.git
   cd Pneumonia-Detection-Using-Machine-Learning
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   
   Download the Chest X-Ray Pneumonia dataset from Kaggle:
   
   [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   
   Extract and organize the data in the following structure:
   ```
   data/
   ├── train/
   │   ├── NORMAL/
   │   ├── PNEUMONIA_BACTERIAL/
   │   └── PNEUMONIA_VIRAL/
   ├── val/
   │   ├── NORMAL/
   │   ├── PNEUMONIA_BACTERIAL/
   │   └── PNEUMONIA_VIRAL/
   └── test/
       ├── NORMAL/
       ├── PNEUMONIA_BACTERIAL/
       └── PNEUMONIA_VIRAL/
   ```

## 📖 Usage

### Training the Model

Train the pneumonia detection model:

```bash
python train.py
```

This will:
- Load and preprocess the training data
- Build the VGG16-based model
- Train with data augmentation
- Save the best model to `models/best_model.h5`
- Generate training history plots

### Evaluating the Model

Evaluate the trained model on test data:

```bash
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on test dataset
- Generate classification report
- Create confusion matrix visualization
- Show sample predictions

### Running the Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

**Features:**
- Upload chest X-ray images
- Provide image URLs for analysis
- Real-time predictions with confidence scores
- Interactive visualization of class probabilities
- Adjustable confidence threshold

##  Project Structure

```
Pneumonia-Detection-Using-Machine-Learning/
│
├── app.py                      # Streamlit web application
├── train.py                    # Model training script
├── evaluate.py                 # Model evaluation script
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
│
├── Lab/                        # Jupyter notebooks
│   ├── 2320030126_code file1.ipynb
│   └── 2320030126_code file2.ipynb
│
├── Project-Pneumonia Detection Using Machine Learning/
│   ├── Abstract-Pneumonia Detection.pdf.pdf
│   └── Poster-Pneumonia Detection.pdf
│
├── data/                       # Dataset (not included in repo)
│   ├── train/
│   ├── val/
│   └── test/
│
└── models/                     # Saved models and plots
    ├── best_model.h5
    ├── pneumonia_model.h5
    ├── training_history.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Data parameters**: Image size, batch size, data paths
- **Model architecture**: Dense layers, dropout rate, activation functions
- **Training settings**: Epochs, learning rate, optimizer
- **Augmentation**: Rotation, zoom, shift ranges
- **Deployment**: Model path, confidence threshold

##  Results

The model achieves excellent performance on the test dataset:

- **Accuracy**: ~95%
- **Precision**: High across all classes
- **Recall**: Effective detection of pneumonia cases
- **F1-Score**: Balanced performance

*Note: Actual results may vary based on dataset and training parameters*

## 🔬 Technical Details

### Data Preprocessing
- Image rescaling (normalization)
- Shear transformation
- Zoom augmentation
- Horizontal flipping
- Rotation and shifting

### Model Training
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Callbacks**: 
  - ModelCheckpoint (save best model)
  - EarlyStopping (prevent overfitting)
  - ReduceLROnPlateau (adaptive learning rate)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

##  Use Cases

- **Medical Screening**: Assist radiologists in preliminary pneumonia detection
- **Research**: Study pneumonia patterns in chest X-rays
- **Education**: Learn about medical image classification
- **Telemedicine**: Remote diagnosis support tool

##  Medical Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Vignesh Reddy**
- GitHub: [@Vigneshreddy99](https://github.com/Vigneshreddy99)

##  Acknowledgments

- Dataset: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- VGG16 Architecture: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- TensorFlow and Keras teams
- Streamlit for the amazing web framework

## References

1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
2. Kermany, D. S., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning.
3. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep neural networks.
