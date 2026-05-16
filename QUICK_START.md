# 🚀 Quick Start Guide

Get up and running with the Pneumonia Detection System in minutes!

## 📋 Prerequisites Checklist

- [ ] Python 3.8 or higher installed
- [ ] pip package manager
- [ ] 5GB free disk space (for dataset and models)
- [ ] Internet connection (for downloading dependencies)

## ⚡ 5-Minute Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vigneshreddy99/Pneumonia-Detection-Using-Machine-Learning.git
cd Pneumonia-Detection-Using-Machine-Learning
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including TensorFlow, Streamlit, and more.

### Step 4: Download Dataset

1. Go to [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Download the dataset (5.8 GB)
3. Extract to the `data/` folder in your project directory

**Expected structure:**
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🎯 Usage Options

### Option 1: Use Pre-trained Model (Fastest)

If you have a pre-trained model file:

1. Place the model file in `models/pneumonia_model.h5`
2. Run the web app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501`
4. Upload an X-ray image and get instant predictions!

### Option 2: Train Your Own Model

**Step 1: Train the model**
```bash
python train.py
```

This will:
- Load and preprocess training data
- Train the VGG16-based model
- Save the best model automatically
- Generate training visualizations

**Training time:** ~2-4 hours (depending on your hardware)

**Step 2: Evaluate the model**
```bash
python evaluate.py
```

This will:
- Test the model on unseen data
- Generate performance metrics
- Create confusion matrix and sample predictions

**Step 3: Run the web application**
```bash
streamlit run app.py
```

## 🖥️ Using the Web Application

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload an X-ray:**
   - Click "Browse files" to upload from your computer
   - Or paste an image URL

3. **Analyze:**
   - Click "Analyze X-Ray" button
   - View prediction results with confidence scores

4. **Interpret results:**
   - Green box = Normal
   - Red box = Pneumonia detected
   - Check confidence percentage
   - Review class probabilities

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Change image size
data:
  img_height: 224
  img_width: 224

# Adjust training parameters
training:
  epochs: 25
  learning_rate: 0.0001

# Set confidence threshold
deployment:
  confidence_threshold: 0.7
```

## 📊 Expected Results

After training, you should see:

- **Training Accuracy:** ~95%
- **Validation Accuracy:** ~92-94%
- **Test Accuracy:** ~90-93%

Results saved in `models/` folder:
- `best_model.h5` - Best performing model
- `training_history.png` - Training curves
- `confusion_matrix.png` - Performance visualization
- `sample_predictions.png` - Example predictions

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "Out of memory" during training

**Solution:**
Reduce batch size in `config.yaml`:
```yaml
data:
  batch_size: 16  # Reduce from 32
```

### Issue: Model file not found

**Solution:**
Make sure you've trained the model first:
```bash
python train.py
```

Or download a pre-trained model and place it in `models/pneumonia_model.h5`

### Issue: Streamlit won't start

**Solution:**
```bash
pip install --upgrade streamlit
streamlit run app.py
```

## 💡 Tips for Best Results

1. **Use high-quality X-ray images** - Clear, well-lit chest X-rays work best
2. **Check confidence scores** - Higher confidence = more reliable prediction
3. **Consult medical professionals** - This tool is for research/education only
4. **Train longer** - More epochs can improve accuracy (but watch for overfitting)
5. **Use data augmentation** - Already configured in `config.yaml`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the Jupyter notebooks in `Lab/` folder
- Check out the research papers in `Project-Pneumonia Detection Using Machine Learning/`
- Customize the model architecture in `train.py`
- Improve the web interface in `app.py`

## 🆘 Need Help?

- **Issues:** Open an issue on [GitHub](https://github.com/Vigneshreddy99/Pneumonia-Detection-Using-Machine-Learning/issues)
- **Questions:** Check existing issues or create a new one
- **Contributions:** Pull requests are welcome!

## ⚠️ Important Reminder

This tool is for **educational and research purposes only**. Always consult qualified healthcare professionals for medical diagnosis and treatment.

---

Happy detecting! 🏥✨
