# Gunakan Python 3.10 slim sebagai base image yang ringan dan efisien
FROM python:3.10-slim

# Konfigurasi environment variables agar Python berjalan optimal di Docker
# PYTHONDONTWRITEBYTECODE=1: Mencegah Python menulis berkas .pyc yang tidak diperlukan
# PYTHONUNBUFFERED=1: Memaksa log output langsung dicetak ke terminal (stdout/stderr) tanpa buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Perbarui paket manajer dan pasang libglib2.0-0 untuk OpenCV Headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin daftar dependensi terlebih dahulu (memanfaatkan Docker Layer Caching)
COPY requirements.txt .

# Instal PyTorch & Torchvision versi CPU untuk menghemat ukuran image (~2 GB lebih hemat)
# lalu pasang dependensi lain dari requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Salin berkas-berkas aplikasi utama, aset statis, dan model PyTorch SOTA
COPY main.py .
COPY database.py .
COPY train_sota_pytorch.py .
COPY index.html .
COPY static/ ./static/
COPY samaya_rafdb_sota_pytorch_b2_adamw.pth .

# Ekspos port 8000 agar dapat diakses dari host machine
EXPOSE 8000

# Jalankan FastAPI menggunakan uvicorn server pada host 0.0.0.0 agar bisa diakses eksternal
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
