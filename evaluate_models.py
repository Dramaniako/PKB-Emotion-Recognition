import os
import argparse
import numpy as np
from tqdm import tqdm

EMOTIONS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

def evaluate_pytorch(model_path, dataset_dir):
    import torch
    import torch.nn.functional as F
    from train_sota_pytorch import build_model, get_balanced_dataset, val_transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Evaluasi PyTorch menggunakan device: {device}")
    
    # Load model
    model = build_model(pretrained_weights=model_path, trainable_backbone=False)
    model = model.to(device)
    model.eval()
    
    # Load data
    try:
        val_loader = get_balanced_dataset(dataset_dir, val_transform, is_training=False)
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return
        
    num_classes = len(EMOTIONS)
    class_correct_std = [0] * num_classes
    class_total_std = [0] * num_classes
    
    class_correct_tta = [0] * num_classes
    class_total_tta = [0] * num_classes
    
    total_std_correct = 0
    total_tta_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Menguji PyTorch Model"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Standard prediction
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=-1)
            _, predicted_std = probs.max(1)
            
            # TTA prediction (Horizontal Flip)
            inputs_flipped = torch.flip(inputs, dims=[3])
            outputs_flipped = model(inputs_flipped)
            probs_flipped = F.softmax(outputs_flipped, dim=-1)
            avg_probs = (probs + probs_flipped) / 2.0
            _, predicted_tta = avg_probs.max(1)
            
            # Accumulate
            for label, pred_std, pred_tta in zip(targets.cpu().numpy(), predicted_std.cpu().numpy(), predicted_tta.cpu().numpy()):
                label = int(label)
                
                class_total_std[label] += 1
                if pred_std == label:
                    class_correct_std[label] += 1
                    total_std_correct += 1
                    
                class_total_tta[label] += 1
                if pred_tta == label:
                    class_correct_tta[label] += 1
                    total_tta_correct += 1
                    
                total_samples += 1
                
    print_report(total_samples, total_std_correct, total_tta_correct, class_correct_std, class_total_std, class_correct_tta, class_total_tta)

def evaluate_keras(model_path, dataset_dir):
    import tensorflow as tf
    from train_sota import get_balanced_dataset as get_keras_dataset
    
    print("[INFO] Evaluasi Keras menggunakan TensorFlow.")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load data
    try:
        val_ds = get_keras_dataset(dataset_dir, is_training=False)
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return
        
    num_classes = len(EMOTIONS)
    class_correct_std = [0] * num_classes
    class_total_std = [0] * num_classes
    
    class_correct_tta = [0] * num_classes
    class_total_tta = [0] * num_classes
    
    total_std_correct = 0
    total_tta_correct = 0
    total_samples = 0
    
    for inputs, targets in tqdm(val_ds, desc="Menguji Keras Model"):
        # inputs: [batch, 224, 224, 3], targets: one-hot [batch, 7]
        targets_idx = np.argmax(targets.numpy(), axis=1)
        
        # Standard prediction
        preds = model.predict(inputs, verbose=0)
        predicted_std = np.argmax(preds, axis=1)
        
        # TTA prediction (Horizontal Flip)
        inputs_flipped = tf.image.flip_left_right(inputs)
        preds_flipped = model.predict(inputs_flipped, verbose=0)
        
        avg_preds = (preds + preds_flipped) / 2.0
        predicted_tta = np.argmax(avg_preds, axis=1)
        
        # Accumulate
        for label, pred_std, pred_tta in zip(targets_idx, predicted_std, predicted_tta):
            label = int(label)
            
            class_total_std[label] += 1
            if pred_std == label:
                class_correct_std[label] += 1
                total_std_correct += 1
                
            class_total_tta[label] += 1
            if pred_tta == label:
                class_correct_tta[label] += 1
                total_tta_correct += 1
                
            total_samples += 1
            
    print_report(total_samples, total_std_correct, total_tta_correct, class_correct_std, class_total_std, class_correct_tta, class_total_tta)

def print_report(total_samples, total_std_correct, total_tta_correct, class_correct_std, class_total_std, class_correct_tta, class_total_tta):
    num_classes = len(EMOTIONS)
    print("\n" + "="*60)
    print("             LAPORAN EVALUASI DETAIL")
    print("="*60)
    print(f"Total Sampel Uji: {total_samples}")
    print(f"Akurasi Keseluruhan (Standar): {100.0 * total_std_correct / total_samples:.2f}%")
    print(f"Akurasi Keseluruhan (Dengan TTA): {100.0 * total_tta_correct / total_samples:.2f}%")
    print("-" * 60)
    print(f"{'Kelas Emosi':<15} | {'Akurasi Standar':<18} | {'Akurasi TTA':<18}")
    print("-" * 60)
    for i in range(num_classes):
        std_acc = (100.0 * class_correct_std[i] / class_total_std[i]) if class_total_std[i] > 0 else 0.0
        tta_acc = (100.0 * class_correct_tta[i] / class_total_tta[i]) if class_total_tta[i] > 0 else 0.0
        c_name = EMOTIONS[i]
        print(f"{c_name:<15} | {std_acc:6.2f}% ({class_correct_std[i]}/{class_total_std[i]})"
              f" | {tta_acc:6.2f}% ({class_correct_tta[i]}/{class_total_tta[i]})")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Script untuk Evaluasi Model PyTorch dan Keras.")
    parser.add_argument('--model-path', type=str, default='samaya_rafdb_sota_pytorch_b2_adamw.pth',
                        help='Path ke file bobot/model (.pth atau .keras)')
    parser.add_argument('--dataset-dir', type=str, default='dataset_rafdb/test',
                        help='Direktori data uji (default: dataset_rafdb/test)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"[ERROR] File model tidak ditemukan di: {args.model_path}")
        return
        
    if not os.path.exists(args.dataset_dir):
        print(f"[ERROR] Direktori dataset tidak ditemukan di: {args.dataset_dir}")
        return
        
    print(f"\n[INFO] Memulai evaluasi untuk model: {args.model_path}")
    print(f"[INFO] Dataset path: {args.dataset_dir}")
    
    if args.model_path.endswith('.pth'):
        evaluate_pytorch(args.model_path, args.dataset_dir)
    elif args.model_path.endswith('.keras'):
        evaluate_keras(args.model_path, args.dataset_dir)
    else:
        print("[ERROR] Format model tidak didukung. Harap gunakan ekstensi .pth (PyTorch) atau .keras (Keras)")

if __name__ == '__main__':
    main()
