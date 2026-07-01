import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: GPU & Environment Setup
cell1 = nbf.v4.new_code_cell("""import tensorflow as tf

# Konfigurasi untuk mendeteksi GPU (khususnya jika berjalan di WSL2 / Colab)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Ditemukan {len(gpus)} GPU(s). Pelatihan akan dipercepat oleh GPU.")
    except RuntimeError as e:
        print(e)
else:
    print("PERINGATAN: GPU tidak ditemukan! Pelatihan akan berjalan di CPU.")
    print("Jika menggunakan Windows secara native, pertimbangkan menggunakan WSL2 untuk dukungan GPU.")""")

# Cell 2: Imports & Configuration
cell2 = nbf.v4.new_code_cell("""import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

IMG_SIZE = 224
BATCH_SIZE = 32 # Menggunakan 32 agar memori GPU/CPU lebih ringan untuk EfficientNet
NUM_CLASSES = 7

# Path Dataset
FER_TRAIN_DIR = 'dataset_fer2013/train'
FER_TEST_DIR = 'dataset_fer2013/test'
RAF_TRAIN_DIR = 'dataset_rafdb/train'
RAF_TEST_DIR = 'dataset_rafdb/test'

print("Konfigurasi selesai.")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")""")

# Cell 3: Balanced Dataset Loader (Oversampling in TF.data)
cell3 = nbf.v4.new_code_cell("""# Fungsi pembantu untuk memuat gambar & label
def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # Catatan: EfficientNet di tf.keras memiliki layer Rescaling internal, 
    # jadi kita kirimkan data RGB skala [0, 255]
    return img, label

def get_balanced_dataset(data_dir, is_training=True):
    class_names = sorted(os.listdir(data_dir))
    paths = []
    labels = []
    
    # List seluruh berkas dan label
    for i, c in enumerate(class_names):
        class_dir = os.path.join(data_dir, c)
        class_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        paths.extend(class_paths)
        labels.extend([i] * len(class_paths))
        
    print(f"=== Dataset dari {data_dir} ===")
    print("Distribusi asli:", Counter(labels))
    
    if is_training:
        # Lakukan Oversampling untuk menyeimbangkan kelas
        counter = Counter(labels)
        max_count = max(counter.values())
        
        balanced_paths = []
        balanced_labels = []
        
        for i in range(len(class_names)):
            class_indices = [idx for idx, label in enumerate(labels) if label == i]
            # Duplikasi index kelas minoritas secara acak agar jumlahnya sama dengan max_count
            duplicated_indices = np.random.choice(class_indices, size=max_count, replace=True)
            
            balanced_paths.extend([paths[idx] for idx in duplicated_indices])
            balanced_labels.extend([labels[idx] for idx in duplicated_indices])
            
        print("Distribusi setelah oversampling:", Counter(balanced_labels))
        final_paths = balanced_paths
        final_labels = balanced_labels
    else:
        # Untuk data validasi, tidak perlu oversampling
        final_paths = paths
        final_labels = labels
        
    # Konversi label ke categorical one-hot encoding
    final_labels_cat = tf.keras.utils.to_categorical(final_labels, num_classes=NUM_CLASSES)
    
    # Buat tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices((final_paths, final_labels_cat))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Caching ke berkas lokal di WSL (sangat cepat, menghemat RAM)
    import hashlib
    path_hash = hashlib.md5(data_dir.encode('utf-8')).hexdigest()
    cache_dir = "/tmp/tf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cache_{path_hash}")
    ds = ds.cache(cache_path)
    
    if is_training:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds

# Uji coba memuat dataset RAF-DB
raf_train_ds = get_balanced_dataset(RAF_TRAIN_DIR, is_training=True)
raf_val_ds = get_balanced_dataset(RAF_TEST_DIR, is_training=False)""")

# Cell 4: Model Builder Utility (EfficientNet-B0)
cell4 = nbf.v4.new_code_cell("""from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom
from tensorflow.keras.models import Model, Sequential

# Layer augmentasi data yang diintegrasikan langsung ke dalam model
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(factor=0.15),
    RandomContrast(factor=0.2),
    RandomZoom(height_factor=0.1, width_factor=0.1)
], name="data_augmentation")

def build_model(pretrained_weights=None, trainable_backbone=False):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    
    # Memuat backbone EfficientNet-B0 (Bobot ImageNet awal)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = trainable_backbone
    
    x = base_model(x, training=trainable_backbone)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    if pretrained_weights:
        print(f"Memuat bobot pretrained dari: {pretrained_weights}")
        # Memuat bobot untuk seluruh layer kecuali layer klasifikasi akhir (Dense)
        # karena urutan label FER-2013 berbeda dengan RAF-DB
        temp_model = Model(inputs=inputs, outputs=predictions)
        temp_model.load_weights(pretrained_weights)
        
        # Pindahkan bobot backbone ke model baru
        model.layers[2].set_weights(temp_model.layers[2].get_weights())
        print("Bobot backbone berhasil ditransfer!")
        
    return model, base_model

model_test, base_test = build_model()
model_test.summary()""")

# Cell 5: Part 1: Pre-training on FER-2013 (Optional/Skip jika sudah dilatih)
cell5 = nbf.v4.new_code_cell("""from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

# Variabel kontrol: Ubah ke True jika Anda ingin menjalankan/mengulang proses pre-training di FER-2013.
# Jika Anda menjalankan di CPU dan memakan waktu terlalu lama, Anda bisa menonaktifkannya
# dan langsung melatih model pada RAF-DB (walau akurasinya mungkin sedikit lebih rendah).
RUN_FER_PRETRAINING = False

fer_weights_path = 'fer2013_efficientnetb0.keras'

if RUN_FER_PRETRAINING:
    print("=== MEMULAI PRE-TRAINING PADA FER-2013 ===")
    
    # Memuat dataset FER-2013
    fer_train_ds = get_balanced_dataset(FER_TRAIN_DIR, is_training=True)
    fer_val_ds = get_balanced_dataset(FER_TEST_DIR, is_training=False)
    
    # Bangun model (Backbone trainable agar bisa mempelajari fitur wajah secara penuh di FER-2013)
    fer_model, fer_base = build_model(trainable_backbone=True)
    
    # Optimizer & LR Schedule
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
    
    history_fer = fer_model.fit(
        fer_train_ds,
        validation_data=fer_val_ds,
        epochs=15,
        callbacks=callbacks
    )
    print("Pre-training FER-2013 selesai dan disimpan.")
else:
    print("Pre-training FER-2013 dilewati.")
    if os.path.exists(fer_weights_path):
        print(f"Berkas bobot {fer_weights_path} ditemukan dan siap digunakan untuk transfer learning.")
    else:
        print(f"PERINGATAN: Berkas {fer_weights_path} tidak ditemukan. Transfer learning akan langsung menggunakan bobot dasar ImageNet.")""")

# Cell 6: Part 2: Fine-Tuning RAF-DB - Phase 1 (Training Custom Head Only)
cell6 = nbf.v4.new_code_cell("""print("=== RAF-DB FINE-TUNING PHASE 1: Training Custom Head Only ===")

# Tentukan jika berkas bobot pretrained FER-2013 ada
weights_to_load = fer_weights_path if os.path.exists(fer_weights_path) else None

# Buat model baru dengan bobot transfer learning. Backbone di-freeze terlebih dahulu.
raf_model, raf_base = build_model(pretrained_weights=weights_to_load, trainable_backbone=False)

# Optimizer & Cosine Decay
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

history_p1 = raf_model.fit(
    raf_train_ds,
    validation_data=raf_val_ds,
    epochs=10,
    callbacks=callbacks
)""")

# Cell 7: Part 2: Fine-Tuning RAF-DB - Phase 2 (Fine-Tuning Top Layers of Backbone)
cell7 = nbf.v4.new_code_cell("""print("=== RAF-DB FINE-TUNING PHASE 2: Fine-Tuning Top 30 Layers of Backbone ===")

# Unfreeze backbone dan set top 30 layer agar bisa di-tune
raf_base.trainable = True
for layer in raf_base.layers[:-30]:
    layer.trainable = False

# Gunakan Learning Rate yang jauh lebih kecil agar tidak merusak representasi fitur
lr_schedule_p2 = CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=steps_per_epoch * 5)

raf_model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p2),
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1),
    metrics=['accuracy']
)

raf_model.summary()

history_p2 = raf_model.fit(
    raf_train_ds,
    validation_data=raf_val_ds,
    epochs=15,
    callbacks=callbacks
)""")

# Cell 8: Part 2: Fine-Tuning RAF-DB - Phase 3 (Fine-Tuning All Layers)
cell8 = nbf.v4.new_code_cell("""print("=== RAF-DB FINE-TUNING PHASE 3: Fine-Tuning Seluruh Model (Unfreeze All) ===")

# Unfreeze seluruh model
raf_base.trainable = True

# Gunakan Learning Rate sangat kecil
lr_schedule_p3 = CosineDecayRestarts(initial_learning_rate=1e-5, first_decay_steps=steps_per_epoch * 5)

raf_model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p3),
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1),
    metrics=['accuracy']
)

raf_model.summary()

history_p3 = raf_model.fit(
    raf_train_ds,
    validation_data=raf_val_ds,
    epochs=10,
    callbacks=callbacks
)""")

# Cell 9: Evaluation
cell9 = nbf.v4.new_code_cell("""import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalFocalCrossentropy

print("Sedang melakukan evaluasi akhir pada model terbaik...")

# Muat model terbaik yang disimpan
best_model = load_model(
    'samaya_rafdb_sota.keras',
    custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy}
)

# Prediksi
predictions = best_model.predict(raf_val_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate([y for x, y in raf_val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

# Ambil nama folder kelas asli sebagai label target
class_names = sorted(os.listdir(RAF_TRAIN_DIR))
# Mapping RAF-DB class names ke emosi yang sesuai
emotion_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

print("\\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=emotion_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=emotion_names, yticklabels=emotion_names)
plt.title('Confusion Matrix - SOTA EfficientNet-B0')
plt.ylabel('Label Asli (True)')
plt.xlabel('Prediksi Model (Predicted)')
plt.tight_layout()
plt.savefig('samaya_rafdb_confusion_matrix_sota.png', dpi=300)
plt.show()""")

nb['cells'] = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

with open('PKB_SAMAYA_RAFDB_SOTA.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Berhasil membuat file PKB_SAMAYA_RAFDB_SOTA.ipynb")
