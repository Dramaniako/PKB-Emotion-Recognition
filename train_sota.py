import os
import random
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

# ----------------- CONFIGURATION -----------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Set to True if you want to pre-train on FER-2013 first
RUN_FER_PRETRAINING = False  

# Dataset paths
FER_TRAIN_DIR = 'dataset_fer2013/train'
FER_TEST_DIR = 'dataset_fer2013/test'
RAF_TRAIN_DIR = 'dataset_rafdb/train'
RAF_TEST_DIR = 'dataset_rafdb/test'
# -------------------------------------------------

# 1. GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Ditemukan {len(gpus)} GPU(s). Pelatihan menggunakan akselerasi GPU.")
    except RuntimeError as e:
        print(f"[WARN] Gagal set memory growth: {e}")
else:
    print("[WARN] GPU tidak terdeteksi oleh TensorFlow! Pelatihan akan berjalan di CPU.")
    print("       Pastikan CUDA Toolkit dan cuDNN sudah terinstal dengan benar di WSL Anda.")

# 2. Balanced Dataset Loader
def load_raw_image(path, label):
    img = tf.io.read_file(path)
    return img, label

def preprocess_raw_image(img_bytes, label):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

def get_balanced_dataset(data_dir, is_training=True):
    class_names = sorted(os.listdir(data_dir))
    paths = []
    labels = []
    
    for i, c in enumerate(class_names):
        class_dir = os.path.join(data_dir, c)
        class_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        paths.extend(class_paths)
        labels.extend([i] * len(class_paths))
        
    print(f"\n[INFO] Dataset {data_dir} - Sampel Asli: {len(paths)}")
    
    if is_training:
        counter = Counter(labels)
        max_count = max(counter.values())
        
        balanced_paths = []
        balanced_labels = []
        
        for i in range(len(class_names)):
            class_indices = [idx for idx, label in enumerate(labels) if label == i]
            duplicated_indices = np.random.choice(class_indices, size=max_count, replace=True)
            
            balanced_paths.extend([paths[idx] for idx in duplicated_indices])
            balanced_labels.extend([labels[idx] for idx in duplicated_indices])
            
        print(f"[INFO] Setelah Oversampling: {len(balanced_paths)} sampel (seimbang)")
        final_paths = balanced_paths
        final_labels = balanced_labels
    else:
        final_paths = paths
        final_labels = labels
        
    final_labels_cat = tf.keras.utils.to_categorical(final_labels, num_classes=NUM_CLASSES)
    
    ds = tf.data.Dataset.from_tensor_slices((final_paths, final_labels_cat))
    
    # 1. Baca bytes mentah dari disk (cepat & hemat memori)
    ds = ds.map(load_raw_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Caching bytes mentah ke RAM (hanya ~100MB RAM, bebas lockfile disk)
    ds = ds.cache()
    
    # 3. Dekode & resize setelah cache agar in-memory cache berukuran kecil
    ds = ds.map(preprocess_raw_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.shuffle(buffer_size=2048) # Menggunakan buffer kecil (2048) agar ramah RAM
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# 3. Model Builder
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(factor=0.15),
    RandomContrast(factor=0.2),
    RandomZoom(height_factor=0.1, width_factor=0.1)
], name="data_augmentation")

def build_model(pretrained_weights=None, trainable_backbone=False):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = trainable_backbone
    
    # Biarkan Keras mengendalikan parameter training secara dinamis
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"[INFO] Memuat bobot kustom dari: {pretrained_weights}")
        model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
        print("[INFO] Bobot berhasil dimuat!")
        
    return model, base_model

# 4. Main Training Pipeline
def main():
    # Bersihkan cache TensorFlow dari run sebelumnya yang mungkin terinterupsi/corrupt
    import shutil
    for cache_path in ["/var/tmp/tf_cache", "/tmp/tf_cache", "D:/tmp/tf_cache", "C:/tmp/tf_cache", "./tmp/tf_cache"]:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print(f"[INFO] Membersihkan cache lama di: {cache_path}")
            except Exception as e:
                print(f"[WARN] Gagal membersihkan cache di {cache_path}: {e}")

    fer_weights_path = 'fer2013_efficientnetb0.keras'
    
    # --- STEP 1: PRE-TRAINING ON FER-2013 ---
    if RUN_FER_PRETRAINING:
        print("\n=== TAHAP 1: PRE-TRAINING PADA FER-2013 ===")
        fer_train_ds = get_balanced_dataset(FER_TRAIN_DIR, is_training=True)
        fer_val_ds = get_balanced_dataset(FER_TEST_DIR, is_training=False)
        
        fer_model, _ = build_model(trainable_backbone=True)
        steps_per_epoch = len(fer_train_ds)
        lr_schedule = CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=steps_per_epoch * 5)
        
        fer_model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        callbacks = [
            ModelCheckpoint(fer_weights_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
        ]
        
        fer_model.fit(fer_train_ds, validation_data=fer_val_ds, epochs=15, callbacks=callbacks)
        print("[INFO] Pre-training FER-2013 Selesai.")
    else:
        print("\n[INFO] Tahap 1 (Pre-training FER-2013) dilewati.")

    # --- STEP 2: FINE-TUNING ON RAF-DB ---
    print("\n=== TAHAP 2: FINE-TUNING PADA RAF-DB ===")
    raf_train_ds = get_balanced_dataset(RAF_TRAIN_DIR, is_training=True)
    raf_val_ds = get_balanced_dataset(RAF_TEST_DIR, is_training=False)
    
    # Load model with pre-trained weights if available
    weights_to_load = fer_weights_path if os.path.exists(fer_weights_path) else None
    raf_model, raf_base = build_model(pretrained_weights=weights_to_load, trainable_backbone=False)
    
    # Phase 2.1: Train Custom Head Only
    print("\n>>> Phase 2.1: Training Custom Head Only (10 Epochs)")
    steps_per_epoch = len(raf_train_ds)
    lr_schedule_p1 = CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=steps_per_epoch * 5)
    raf_model.compile(
        optimizer=Adam(learning_rate=lr_schedule_p1),
        loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint('samaya_rafdb_sota.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    ]
    
    raf_model.fit(raf_train_ds, validation_data=raf_val_ds, epochs=10, callbacks=callbacks)
    
    # Phase 2.2: Fine-Tuning Top 30 Layers of Backbone
    print("\n>>> Phase 2.2: Fine-Tuning Top 30 Layers of Backbone (15 Epochs)")
    raf_base.trainable = True
    for layer in raf_base.layers[:-30]:
        layer.trainable = False
        
    lr_schedule_p2 = CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=steps_per_epoch * 5)
    raf_model.compile(
        optimizer=Adam(learning_rate=lr_schedule_p2),
        loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy']
    )
    raf_model.fit(raf_train_ds, validation_data=raf_val_ds, epochs=15, callbacks=callbacks)
    
    # Phase 2.3: Fine-Tuning All Layers
    print("\n>>> Phase 2.3: Fine-Tuning Seluruh Model (10 Epochs)")
    raf_base.trainable = True
    lr_schedule_p3 = CosineDecayRestarts(initial_learning_rate=1e-5, first_decay_steps=steps_per_epoch * 5)
    raf_model.compile(
        optimizer=Adam(learning_rate=lr_schedule_p3),
        loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy']
    )
    raf_model.fit(raf_train_ds, validation_data=raf_val_ds, epochs=10, callbacks=callbacks)
    print("\n[SUCCESS] Proses Fine-Tuning SOTA Model Selesai! Model terbaik disimpan di: 'samaya_rafdb_sota.keras'")

if __name__ == '__main__':
    main()
