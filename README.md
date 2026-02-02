# 🧠 MetaTune: Dataset-Aware Optimization

**MetaTune** is an advanced AI-driven hyperparameter optimization system that uses meta-learning and bilevel optimization to adaptively tune neural networks based on the unique "DNA" of your dataset.

Designed with a futuristic "Out of World" interface, MetaTune bridges the gap between complex machine learning operations and intuitive user experience.

---

## 🚀 Key Features

### 1. 🧬 Forensic Data Diagnosis
Instantly scans your dataset to extract its "DNA":
- **Shape Analysis**: Instances vs. Features.
- **Complexity Score**: Measures target entropy to gauge difficulty.
- **Imbalance Detection**: Identifies class distribution issues.
- **Stability Check**: Automatically flags high-entropy or noisy datasets.

### 2. 🧠 Neural Hyperparameter Prediction
A trained **Meta-Learner** (The Brain) analyzes the dataset's DNA and predicts the optimal training configuration:
- Learning Rate
- Regularization (L2 / Weight Decay)
- Batch Size
- Optimizer Type (Adam, SGD, etc.)

### 3. 🚀 Dynamic Bilevel Training
Executes a training engine that adapts in real-time. The system uses the predicted parameters to train a model, providing live feedback on loss and accuracy.

### 4. 🔄 Cognitive Feedback Loop
MetaTune learns from every run. The performance metrics of the trained model are fed back into the Meta-Learner, continuously improving its prediction accuracy for future datasets.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/metatune.git
   cd MetaTune
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### Option 1: Web Interface (Streamlit)
The recommended way to use MetaTune is via the futuristic web dashboard.

1. **Run the App**
   ```bash
   streamlit run app.py
   ```
2. **Interact**
   - Upload your CSV dataset.
   - Watch the **Forensic Diagnosis** analyze your data.
   - Click **"Query Meta-Learner"** to generate optimal parameters.
   - Click **"Start Training Engine"** to begin the training process.
   - Download the trained model (`.pth`) upon completion.

### Option 2: CLI Pipeline
For automated workflows or power users, use the command-line interface.

```bash
python pipeline.py path/to/dataset.csv --target target_column_name --epochs 20
```

**Arguments:**
- `data`: Path to your CSV dataset.
- `--target`: Name of the target column (optional, will attempt to auto-detect).
- `--epochs`: Number of training epochs (default: 20).

---

## 📂 Project Structure

```
MetaTune/
├── app.py                 # Main Streamlit Web Application
├── pipeline.py            # CLI Entry Point for automation
├── requirements.txt       # Project Dependencies
├── brain.py               # MetaLearner Logic (The Brain)
├── data_analyzer.py       # Dataset Forensics & DNA Extraction
├── engine.py              # Dynamic Training Engine
├── engine_stream.py       # Streaming Engine for Real-time feedback
├── bilevel.py             # Bilevel Optimization Logic
├── global_car_dataset.csv # Example Dataset
└── ...
```

---

## 🔧 Technologies

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Core ML**: [PyTorch](https://pytorch.org/) & [Scikit-Learn](https://scikit-learn.org/)
- **Data Processing**: Pandas & NumPy
- **Tracking**: [Weights & Biases](https://wandb.ai/) (Optional)

---

## 📄 License

This project is open-source and available under the **MIT License**.
