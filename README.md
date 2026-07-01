# SAMAYA: State-of-the-Art Emotion Recognition & Engagement Analyzer

SAMAYA is a real-time emotion recognition and user engagement scoring system built for Edge AI applications. It leverages a modern deep learning architecture (**EfficientNet-B2** trained with **Categorical Focal Loss**, **Label Smoothing**, and **Test-Time Augmentation (TTA)**) to achieve state-of-the-art (SOTA) performance of **80.15% validation accuracy** on the RAF-DB dataset.

The system features:
- **FastAPI Backend**: High-performance HTTP server for low-latency image inference.
- **Interactive Web Interface**: Single-page application for real-time camera inference, metrics visualization, and session logs.
- **SQLite Database**: Local logging of emotion frequencies and engagement scores for analysis.
- **Dual-Model Support**: Automatic loading of the best PyTorch SOTA model, with a robust fallback to Keras models.

---

## 📂 Project Structure

```text
├── docs/                     # Project documentation & chapter reports
├── notebooks/                # Jupyter Notebooks for training (optimized for Google Colab)
├── models_archive/           # Archived legacy weights and pre-trained backbones
├── static/                   # Frontend assets (CSS, JS, libraries)
├── Dockerfile                # Docker build configuration
├── database.py               # SQLite logger configuration
├── main.py                   # FastAPI server logic and inference handler
├── test_and_evaluate.py      # Script to calculate full metrics (Acc, F1, Precision, Recall, CM)
├── evaluate_models.py        # Custom evaluation script
├── index.html                # Main single-page web interface
├── requirements.txt          # Python packages list
└── samaya_rafdb_sota_pytorch_b2_adamw.pth  # SOTA PyTorch Model weights (80.15% Acc)
```

---

## ⚡ Quick Start (Local Run)

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Create a virtual environment and install the required dependencies:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
Start the FastAPI server:
```bash
python main.py
```
Or use the provided batch script (Windows):
```cmd
jalankan_samaya.bat
```
Open your browser and navigate to **`http://localhost:8000`**.

---

## 🐳 Running with Docker

You can easily run the entire system inside a Docker container without needing to configure a local Python environment or install graphics/CUDA dependencies.

### 1. Build the Docker Image
Run the following command in the project root directory:
```bash
docker build -t samaya-app .
```

### 2. Run the Container (Standard Mode)
Run the built container and map port 8000:
```bash
docker run -d -p 8000:8000 --name samaya-container samaya-app
```

### 3. Run the Container (With Database Persistence - Recommended)
To prevent your engagement logs and database from being lost when the container stops, mount the SQLite database file from your host machine:

**On Windows (PowerShell):**
```powershell
docker run -d -p 8000:8000 -v ${PWD}/samaya_logs.db:/app/samaya_logs.db --name samaya-container samaya-app
```

**On Linux/macOS:**
```bash
docker run -d -p 8000:8000 -v $(pwd)/samaya_logs.db:/app/samaya_logs.db --name samaya-container samaya-app
```

Now, navigate to **`http://localhost:8000`** in your browser to use the application. All database entries will be persisted in your local workspace.

### 4. Stopping and Managing the Container
```bash
# View running containers
docker ps

# Stop the container
docker stop samaya-container

# Start it again
docker start samaya-container

# Remove the container
docker rm -f samaya-container
```

---

## 📊 Model Evaluation

To run a complete evaluation report (generating accuracy, precision, recall, F1-scores, and a text/graphical confusion matrix):

```bash
# Install plotting libraries in your environment
pip install matplotlib seaborn

# Run the evaluation script
python test_and_evaluate.py
```
This will print a complete classification report to the console and save the graphical confusion matrix to `samaya_rafdb_confusion_matrix_sota.png`.

---

## 🧠 Training & Colab Notebooks
To train or fine-tune models from scratch:
1. Open the [notebooks/PKB_SAMAYA_RAFDB_SOTA_Colab.ipynb](file:///d:/Project/PKB-Emotion%20Recognition/notebooks/PKB_SAMAYA_RAFDB_SOTA_Colab.ipynb) in Google Colab.
2. Mount your Google Drive to save checkpoints.
3. Once training is complete, download `samaya_rafdb_sota_pytorch_b2_adamw.pth` and place it in the project root directory.

---

## ✉️ License & Contact
This project is developed as part of a research initiative on PKB (Pola Kognitif & Berpikir) Emotion Recognition. For questions, please refer to the documents in the `docs/` folder.
