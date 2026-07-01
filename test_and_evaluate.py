import os
import argparse
import numpy as np
from tqdm import tqdm

# Urutan emosi harus sesuai dengan urutan folder/kelas dataset
EMOTIONS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

def evaluate_pytorch(model_path, dataset_dir, use_tta=True):
    import torch
    import torch.nn.functional as F
    from train_sota_pytorch import build_model, get_balanced_dataset, val_transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Evaluasi PyTorch menggunakan device: {device}")
    
    model = build_model(pretrained_weights=model_path, trainable_backbone=False)
    model = model.to(device)
    model.eval()
    
    try:
        val_loader = get_balanced_dataset(dataset_dir, val_transform, is_training=False)
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return None, None
        
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Menginferensi Model PyTorch"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=-1)
            
            if use_tta:
                # TTA: Average prediction dari citra asli + citra yang di-flip horizontal
                inputs_flipped = torch.flip(inputs, dims=[3])
                outputs_flipped = model(inputs_flipped)
                probs_flipped = F.softmax(outputs_flipped, dim=-1)
                
                avg_probs = (probs + probs_flipped) / 2.0
                _, predicted = avg_probs.max(1)
            else:
                _, predicted = probs.max(1)
                
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    return np.array(y_true), np.array(y_pred)

def evaluate_keras(model_path, dataset_dir, use_tta=True):
    import tensorflow as tf
    from train_sota import get_balanced_dataset as get_keras_dataset
    
    print("[INFO] Evaluasi Keras menggunakan TensorFlow")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    try:
        val_ds = get_keras_dataset(dataset_dir, is_training=False)
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return None, None
        
    y_true = []
    y_pred = []
    
    for inputs, targets in tqdm(val_ds, desc="Menginferensi Model Keras"):
        targets_idx = np.argmax(targets.numpy(), axis=1)
        
        preds = model.predict(inputs, verbose=0)
        
        if use_tta:
            inputs_flipped = tf.image.flip_left_right(inputs)
            preds_flipped = model.predict(inputs_flipped, verbose=0)
            avg_preds = (preds + preds_flipped) / 2.0
            predicted = np.argmax(avg_preds, axis=1)
        else:
            predicted = np.argmax(preds, axis=1)
            
        y_true.extend(targets_idx)
        y_pred.extend(predicted)
        
    return np.array(y_true), np.array(y_pred)

def calculate_metrics_pure_python(y_true, y_pred):
    num_classes = len(EMOTIONS)
    
    # 1. Hitung Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # 2. Hitung Metrik per Kelas (Precision, Recall, F1)
    class_metrics = {}
    total_samples = len(y_true)
    total_correct = np.sum(y_true == y_pred)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
        
    # 3. Hitung Rata-rata (Macro & Weighted Avg)
    macro_precision = np.mean([class_metrics[i]['precision'] for i in range(num_classes)])
    macro_recall = np.mean([class_metrics[i]['recall'] for i in range(num_classes)])
    macro_f1 = np.mean([class_metrics[i]['f1-score'] for i in range(num_classes)])
    
    weighted_precision = sum(class_metrics[i]['precision'] * class_metrics[i]['support'] for i in range(num_classes)) / total_samples
    weighted_recall = sum(class_metrics[i]['recall'] * class_metrics[i]['support'] for i in range(num_classes)) / total_samples
    weighted_f1 = sum(class_metrics[i]['f1-score'] * class_metrics[i]['support'] for i in range(num_classes)) / total_samples
    
    # Cetak Classification Report
    print("\n" + "=" * 65)
    print("                    CLASSIFICATION REPORT")
    print("=" * 65)
    print(f"{'Kelas Emosi':<15} | {'precision':<10} | {'recall':<10} | {'f1-score':<10} | {'support':<8}")
    print("-" * 65)
    for i in range(num_classes):
        m = class_metrics[i]
        print(f"{EMOTIONS[i]:<15} | {m['precision']:<10.2f} | {m['recall']:<10.2f} | {m['f1-score']:<10.2f} | {m['support']:<8}")
    print("-" * 65)
    print(f"{'accuracy':<15} | {'':<10} | {'':<10} | {accuracy:<10.2f} | {total_samples:<8}")
    print(f"{'macro avg':<15} | {macro_precision:<10.2f} | {macro_recall:<10.2f} | {macro_f1:<10.2f} | {total_samples:<8}")
    print(f"{'weighted avg':<15} | {weighted_precision:<10.2f} | {weighted_recall:<10.2f} | {weighted_f1:<10.2f} | {total_samples:<8}")
    print("=" * 65)
    
    # Cetak Confusion Matrix dalam bentuk teks
    print("\n" + "=" * 65)
    print("                CONFUSION MATRIX (Teks)")
    print("=" * 65)
    print(f"{'True vs Pred':<12}", end="")
    for e in EMOTIONS:
        print(f"{e[:4]:>7}", end="")
    print("\n" + "-" * 65)
    for i in range(num_classes):
        print(f"{EMOTIONS[i]:<12}", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:>7}", end="")
        print()
    print("=" * 65)
    
    return cm

def plot_confusion_matrix(cm):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
        plt.title('Confusion Matrix - SOTA Model')
        plt.ylabel('Label Asli (True)')
        plt.xlabel('Prediksi Model (Predicted)')
        plt.tight_layout()
        
        output_path = 'samaya_rafdb_confusion_matrix_sota.png'
        plt.savefig(output_path, dpi=300)
        print(f"\n[SUCCESS] Gambar Confusion Matrix berhasil disimpan ke: '{output_path}'")
        plt.close()
    except ImportError:
        print("\n[INFO] Matplotlib atau Seaborn tidak terinstal di lingkungan lokal.")
        print("[INFO] Jalankan perintah berikut untuk mengaktifkan plot grafis:")
        print("       pip install matplotlib seaborn")

def main():
    parser = argparse.ArgumentParser(description="Script Pengujian Model SOTA (Accuracy, F1-Score, Confusion Matrix).")
    parser.add_argument('--model-path', type=str, default='samaya_rafdb_sota_pytorch_b2_adamw.pth',
                        help='Path ke file bobot model (.pth atau .keras)')
    parser.add_argument('--dataset-dir', type=str, default='dataset_rafdb/test',
                        help='Direktori dataset uji (default: dataset_rafdb/test)')
    parser.add_argument('--no-tta', action='store_true', help='Matikan Test-Time Augmentation (TTA)')
    
    args = parser.parse_args()
    use_tta = not args.no_tta
    
    if not os.path.exists(args.model_path):
        print(f"[ERROR] File model tidak ditemukan: {args.model_path}")
        return
        
    if not os.path.exists(args.dataset_dir):
        print(f"[ERROR] Direktori dataset tidak ditemukan: {args.dataset_dir}")
        return
        
    print(f"\n[INFO] Memulai pengujian lengkap untuk: {args.model_path}")
    print(f"[INFO] Jalur dataset: {args.dataset_dir}")
    print(f"[INFO] Menggunakan TTA: {use_tta}")
    
    if args.model_path.endswith('.pth'):
        y_true, y_pred = evaluate_pytorch(args.model_path, args.dataset_dir, use_tta)
    elif args.model_path.endswith('.keras'):
        y_true, y_pred = evaluate_keras(args.model_path, args.dataset_dir, use_tta)
    else:
        print("[ERROR] Format file tidak dikenal. Harap gunakan file .pth atau .keras")
        return
        
    if y_true is not None and y_pred is not None:
        cm = calculate_metrics_pure_python(y_true, y_pred)
        plot_confusion_matrix(cm)

if __name__ == '__main__':
    main()
