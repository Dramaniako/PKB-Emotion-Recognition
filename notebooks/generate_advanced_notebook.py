import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: GPU Setup
cell1 = nbf.v4.new_code_cell("""import tensorflow as tf

# Konfigurasi untuk menggunakan GPU Lokal
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Mengaktifkan memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Ditemukan {len(gpus)} GPU(s) lokal.")
    except RuntimeError as e:
        print(e)
else:
    print("PERINGATAN: GPU tidak ditemukan! Pelatihan akan berjalan di CPU.")""")

# Cell 2: Imports, Config, Class Weights
cell2 = nbf.v4.new_code_cell("""import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 7
TRAIN_DIR = 'dataset_rafdb/train'
TEST_DIR = 'dataset_rafdb/test'

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical', shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False
)

class_names = sorted(os.listdir(TRAIN_DIR))
y_train = []
for i, c in enumerate(class_names):
    count = len(os.listdir(os.path.join(TRAIN_DIR, c)))
    y_train.extend([i] * count)
y_train = np.array(y_train)

class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights_arr)}

print("Class Names:", class_names)
print("Class Weights:", class_weight_dict)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)""")

# Cell 3: Model Architecture
cell3 = nbf.v4.new_code_cell("""from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom, Rescaling
from tensorflow.keras.models import Model, Sequential

data_augmentation = Sequential([
    Rescaling(1./255),
    RandomFlip("horizontal"),
    RandomRotation(factor=0.15),
    RandomContrast(factor=0.2),
    RandomZoom(height_factor=0.1, width_factor=0.1)
], name="data_augmentation")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False # Freeze the base for Phase 1

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.summary()""")

# Cell 4: Losses and Callbacks
cell4 = nbf.v4.new_code_cell("""from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalFocalCrossentropy

# Advanced Optimizations: Focal Loss + Label Smoothing
# Focal loss memaksa model fokus pada 'hard examples' (kelas minoritas yang sulit).
# Label smoothing=0.1 membuat target probabilitas tidak mutlak 1.0/0.0, mencegah overconfidence.
loss_fn = CategoricalFocalCrossentropy(
    alpha=0.25, 
    gamma=2.0, 
    label_smoothing=0.1
)

# Callback (tanpa ReduceLROnPlateau karena kita akan pakai Cosine Decay di Optimizer)
callbacks = [
    ModelCheckpoint('samaya_rafdb_advanced.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
]""")

# Cell 5: Phase 1
cell5 = nbf.v4.new_code_cell("""from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

print("=== PHASE 1: Training Custom Head Only ===")
EPOCHS_PHASE_1 = 10
steps_per_epoch = len(train_ds)

# Cosine Decay Restarts Schedule
lr_schedule_p1 = CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=steps_per_epoch * 5 # Restart setiap 5 epoch
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p1),
    loss=loss_fn,
    metrics=['accuracy']
)

history_phase1 = model.fit(
    train_ds, 
    validation_data=val_ds,
    epochs=EPOCHS_PHASE_1,
    callbacks=callbacks,
    class_weight=class_weight_dict
)""")

# Cell 6: Phase 2
cell6 = nbf.v4.new_code_cell("""print("=== PHASE 2: Fine-Tuning Top 30 Layers of Base Model ===")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

EPOCHS_PHASE_2 = 20
steps_per_epoch = len(train_ds)

# Learning rate yang jauh lebih kecil untuk fine-tuning
lr_schedule_p2 = CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=steps_per_epoch * 5
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p2),
    loss=loss_fn,
    metrics=['accuracy']
)

model.summary()

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE_2,
    callbacks=callbacks,
    class_weight=class_weight_dict
)""")

# Cell 7: Phase 3
cell7 = nbf.v4.new_code_cell("""print("=== PHASE 3: Fine-Tuning Seluruh Base Model (Unfreeze All) ===")

# Unfreeze SELURUH layer
base_model.trainable = True

EPOCHS_PHASE_3 = 15

# Learning rate SANGAT kecil agar tidak merusak bobot pre-trained yang sudah disesuaikan
lr_schedule_p3 = CosineDecayRestarts(
    initial_learning_rate=1e-5,
    first_decay_steps=steps_per_epoch * 5
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p3),
    loss=loss_fn,
    metrics=['accuracy']
)

model.summary()

history_phase3 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE_3,
    callbacks=callbacks,
    class_weight=class_weight_dict
)""")

# Cell 8: Evaluation
cell8 = nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

print("Sedang melakukan inferensi pada data validasi dengan model terbaik...")
# Muat model terbaik yang disimpan (akan memiliki custom loss)
# Jika ada error saat load model karena custom loss, sertakan custom_objects
best_model = load_model(
    'samaya_rafdb_advanced.keras', 
    custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy}
)

predictions = best_model.predict(val_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

print("\\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Advanced Optimized MobileNetV2')
plt.ylabel('Label Asli (True)')
plt.xlabel('Prediksi Model (Predicted)')
plt.tight_layout()
plt.savefig('samaya_rafdb_confusion_matrix_advanced.png', dpi=300)
plt.show()""")

nb['cells'] = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]

with open('PKB_SAMAYA_RAFDB_Advanced_Optimized.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Berhasil membuat file PKB_SAMAYA_RAFDB_Advanced_Optimized.ipynb")
