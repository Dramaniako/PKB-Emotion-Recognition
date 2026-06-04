# Gunakan Python 3.10 slim sebagai base image yang ringan dan efisien
FROM python:3.10-slim

# Konfigurasi environment variables agar Python berjalan optimal di Docker
# PYTHONDONTWRITEBYTECODE=1: Mencegah Python menulis berkas .pyc yang tidak diperlukan
# PYTHONUNBUFFERED=1: Memaksa log output langsung dicetak ke terminal (stdout/stderr) tanpa buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Perbarui paket manajer dan pasang build-essential (untuk kemudahan kompilasi jika diperlukan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin daftar dependensi terlebih dahulu (memanfaatkan Docker Layer Caching)
COPY requirements.txt .

# Instal seluruh dependensi Python tanpa menyimpan cache (menghemat ruang disk)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin berkas-berkas aplikasi utama
COPY main.py .
COPY database.py .
COPY index.html .
COPY samaya_rafdb_advanced.keras .

# Ekspos port 8000 agar dapat diakses dari host machine
EXPOSE 8000

# Jalankan FastAPI menggunakan uvicorn server pada host 0.0.0.0 agar bisa diakses eksternal
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
