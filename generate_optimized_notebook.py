import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: GPU Setup
cell1 = nbf.v4.new_code_cell("""import tensorflow as tf

# Konfigurasi untuk menggunakan GPU Lokal
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Mengaktifkan memory growth agar TensorFlow tidak memakan seluruh VRAM secara langsung
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Ditemukan {len(gpus)} GPU(s) lokal. Pelatihan akan menggunakan GPU.")
    except RuntimeError as e:
        print(e)
else:
    print("PERINGATAN: GPU tidak ditemukan! Pastikan CUDA dan cuDNN sudah terinstal. Pelatihan akan berjalan di CPU.")""")

# Cell 2: Imports, Config, Class Weights
cell2 = nbf.v4.new_code_cell("""import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224 # Diperbesar dari 96 ke 224 sesuai spesifikasi
BATCH_SIZE = 64
NUM_CLASSES = 7
TRAIN_DIR = 'dataset_rafdb/train'
TEST_DIR = 'dataset_rafdb/test'

# 1. Pipeline Training
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)

# 2. Pipeline Validasi
val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Hitung Class Weights untuk menangani Imbalance
class_names = sorted(os.listdir(TRAIN_DIR))
y_train = []
for i, c in enumerate(class_names):
    count = len(os.listdir(os.path.join(TRAIN_DIR, c)))
    y_train.extend([i] * count)
y_train = np.array(y_train)

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights_arr)}

print("Class Names:", class_names)
print("Class Weights yang dihitung:", class_weight_dict)

# Optimasi CPU-to-GPU Handoff
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)""")

# Cell 3: Optional Oversampling strategy
cell3 = nbf.v4.new_code_cell("""# --- ADVANCED: tf.data Random Oversampling (Rejection Resample) ---
# JIKA ANDA INGIN MENGGUNAKAN OVERSAMPLING ALIH-ALIH CLASS WEIGHTS, BACA BAGIAN INI.
# Karena penggunaan Class Weights dan Oversampling sekaligus bisa menyebabkan Over-Correction, 
# kita default ke class weights yang lebih efisien di Keras. Namun, jika ingin mencoba oversampling:

'''
class_counts = [len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names]
initial_dist = [count / len(y_train) for count in class_counts]
target_dist = [1.0/NUM_CLASSES] * NUM_CLASSES

resampler = tf.data.experimental.rejection_resample(
    class_func=lambda x, y: tf.argmax(y, axis=0),
    target_dist=target_dist,
    initial_dist=initial_dist
)

train_ds_unbatched = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=None, label_mode='categorical', shuffle=True
)

resampled_train_ds = train_ds_unbatched.apply(resampler).map(
    lambda class_val, data: data
).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
'''
# Jika ingin pakai `resampled_train_ds`, gunakan itu di model.fit dan hapus argumen `class_weight`.
""")

# Cell 4: Model with Augmentation
cell4 = nbf.v4.new_code_cell("""from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom, Rescaling
from tensorflow.keras.models import Model, Sequential

# 0. Enhanced Data Augmentation (Dieksekusi oleh GPU)
data_augmentation = Sequential([
    Rescaling(1./255),
    RandomFlip("horizontal"),
    RandomRotation(factor=0.15),
    RandomContrast(factor=0.2),
    RandomZoom(height_factor=0.1, width_factor=0.1)
], name="data_augmentation")

# 1. Load Pre-trained Base Model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base for Phase 1
base_model.trainable = False

# 2. Bangun Custom Classification Head dengan Augmentasi
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False) # Pastikan BatchNormalization layer berjalan dalam inference mode (sangat penting untuk fine-tuning)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Tambahkan Dropout 0.5 untuk mencegah overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# 3. Satukan menjadi model utuh
model = Model(inputs=inputs, outputs=predictions)
model.summary()""")

# Cell 5: Callbacks
cell5 = nbf.v4.new_code_cell("""from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint('samaya_rafdb_mobilenetv2_optimized.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    # Early stopping dengan patience 8
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    # ReduceLROnPlateau jika validasi loss tidak turun dalam 3 epoch
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]""")

# Cell 6: Phase 1 Training
cell6 = nbf.v4.new_code_cell("""from tensorflow.keras.optimizers import Adam

print("=== PHASE 1: Training Custom Head Only ===")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS_PHASE_1 = 10

# Melatih bagian head model
history_phase1 = model.fit(
    train_ds, 
    validation_data=val_ds,
    epochs=EPOCHS_PHASE_1,
    callbacks=callbacks,
    class_weight=class_weight_dict # Kita pakai class_weight untuk koreksi imbalance
)""")

# Cell 7: Phase 2 Fine Tuning
cell7 = nbf.v4.new_code_cell("""print("=== PHASE 2: Fine-Tuning Top 30 Layers of Base Model ===")

# Unfreeze base model
base_model.trainable = True

# Freeze all layers except the top 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compile lagi dengan learning rate yang JAUH lebih rendah (1e-5)
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS_PHASE_2 = 30 # Bebas disesuaikan, early stopping akan menghentikannya jika sudah overfit

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE_2,
    callbacks=callbacks,
    class_weight=class_weight_dict # Tetap pakai class weight
)""")

# Cell 8: Evaluation
cell8 = nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

print("Sedang melakukan inferensi pada data validasi dengan model terbaik...")
# Muat model terbaik yang disimpan
best_model = load_model('samaya_rafdb_mobilenetv2_optimized.keras')

predictions = best_model.predict(val_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Optimized MobileNetV2 (Phase 1+2 & Imbalance Checked)')
plt.ylabel('Label Asli (True)')
plt.xlabel('Prediksi Model (Predicted)')
plt.tight_layout()
plt.savefig('samaya_rafdb_confusion_matrix_optimized.png', dpi=300)
plt.show()""")


nb['cells'] = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]

with open('PKB_SAMAYA_RAFDB_Optimized.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Berhasil membuat file PKB_SAMAYA_RAFDB_Optimized.ipynb")
