import os
import argparse
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, cohen_kappa_score, log_loss
)
from sklearn.preprocessing import label_binarize
from autogluon.multimodal import MultiModalPredictor

# ===========================
# 1. Configuration & Argument Parsing
# ===========================

def parse_args():
    parser = argparse.ArgumentParser(description="Training with Absolute Paths")
    
    # --- Path Parameters ---
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='Full path to the CSV data file (e.g., /home/user/data/image_224_all.csv)')
    
    parser.add_argument('--image_root', type=str, default=None, 
                        help='Root directory for image files.')
    
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Full path to the root directory for output results (e.g., /home/user/results/)')
    
    # --- Model Parameters ---
    parser.add_argument('--model_name', type=str, default="vgg19_bn", help='timm model name')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    
    # --- Columns & Proxy ---
    parser.add_argument('--image_col', type=str, default='image', help='Column name for image paths in CSV')
    parser.add_argument('--label_col', type=str, default='label', help='Column name for labels in CSV')
    parser.add_argument('--use_proxy', action='store_true', help='Whether to enable network proxy')
    parser.add_argument('--proxy_url', type=str, default="http://172.19.64.1:7890", help='Proxy URL')

    args = parser.parse_args()
    
    # --- Force conversion to absolute paths ---
    args.csv_path = os.path.abspath(args.csv_path)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Default image_root to CSV directory if not specified
    if args.image_root is None:
        args.image_root = os.path.dirname(args.csv_path)
    else:
        args.image_root = os.path.abspath(args.image_root)

    return args

def setup_env(args):
    warnings.filterwarnings('ignore')
    np.random.seed(args.seed)
    
    # Print path information for verification
    print(f"{'='*10} Path Configuration {'='*10}")
    print(f"CSV File:    {args.csv_path}")
    print(f"Image Root:  {args.image_root}")
    print(f"Output Dir:  {args.output_dir}")
    print(f"{'='*40}")

    if args.use_proxy:
        os.environ["http_proxy"] = args.proxy_url
        os.environ["https_proxy"] = args.proxy_url

def path_expander(path_str, base_folder):
    """Converts relative paths to absolute paths by joining with base_folder"""
    if pd.isna(path_str): return path_str
    path_l = path_str.split(';')
    # Join with base_folder and ensure the result is an absolute path
    return ';'.join([os.path.abspath(os.path.join(base_folder, p.strip())) for p in path_l])

# ===========================
# 2. Data Loading
# ===========================

def load_and_preprocess_data(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    all_data = pd.read_csv(args.csv_path)
    
    # Filter training data
    train_info = all_data[all_data['data_type'] == 'train']
    df = train_info[[args.image_col, args.label_col]].copy()

    # Path processing
    # 1. Take the first path before the semicolon (if multiple paths exist)
    df[args.image_col] = df[args.image_col].apply(lambda ele: ele.split(';')[0])
    # 2. Expand to absolute path based on image_root
    df[args.image_col] = df[args.image_col].apply(lambda ele: path_expander(ele, base_folder=args.image_root))
    
    # Simple validation: Check if the first image exists to catch path configuration errors
    first_img_path = df.iloc[0][args.image_col]
    if not os.path.exists(first_img_path.split(';')[0]):
        print(f"Warning: First image path does not exist: {first_img_path}")
        print(f"Please verify if the --image_root parameter is correct: {args.image_root}")
    
    return df

# ===========================
# 3. Metric Calculation
# ===========================

def calculate_metrics(y_true, y_pred, y_pred_proba):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if isinstance(y_pred_proba, pd.DataFrame):
        y_pred_proba = y_pred_proba.values
    
    metrics = {
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
    
    # Calculate Precision, Recall, and F1 for various averaging methods
    for avg in [None, 'macro', 'micro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        suffix = f"_{avg}" if avg else "_per_class"
        if avg is None:
            metrics[f'precision{suffix}'] = p; metrics[f'recall{suffix}'] = r; metrics[f'f1{suffix}'] = f
        else:
            metrics[f'{avg}_precision'] = p; metrics[f'{avg}_recall'] = r; metrics[f'{avg}_f1'] = f

    # Calculate ROC-AUC and Average Precision (One-vs-Rest)
    n_classes = y_pred_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    try:
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
        metrics['average_precision'] = average_precision_score(y_true_bin, y_pred_proba, average='macro')
    except:
        metrics['roc_auc'] = np.nan; metrics['average_precision'] = np.nan
        
    return metrics

# ===========================
# 4. Training Pipeline
# ===========================

def run_kfold_training(args, df):
    # Construct experiment folder name
    csv_name = os.path.basename(args.csv_path).replace('.csv', '')
    exp_name = f"{csv_name}_{args.model_name}_ep{args.epochs}_pat{args.patience}_{args.k_folds}fold"
    
    # Final results path = output_dir / exp_name
    results_folder = os.path.join(args.output_dir, exp_name)
    os.makedirs(results_folder, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(results_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    hyperparameters = {
        "optimization.max_epochs": args.epochs,
        "optimization.patience": args.patience,
        "optimization.val_check_interval": 1.0,
        "optimization.learning_rate": args.lr,
        "optimization.lr_decay": 0.98,
        "env.batch_size": args.batch_size,
        "model.timm_image.checkpoint_name": args.model_name,
    }

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    all_metrics = []
    pdf = PdfPages(os.path.join(results_folder, 'confusion_matrices.pdf'))

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        print(f"\n>>> Processing Fold {fold}/{args.k_folds}...")
        
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        fold_save_path = os.path.join(results_folder, f'fold_{fold}')
        
        predictor = MultiModalPredictor(label=args.label_col, path=fold_save_path, verbosity=2)
        predictor.fit(
            train_data=train_fold,
            tuning_data=val_fold,
            hyperparameters=hyperparameters,
            standalone=True
        )
        
        y_pred = predictor.predict(val_fold)
        y_pred_proba = predictor.predict_proba(val_fold)
        y_true = val_fold[args.label_col]
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        metrics['fold'] = fold
        all_metrics.append(metrics)
        
        # Save predictions
        res_df = pd.DataFrame({'true': y_true, 'pred': y_pred})
        if isinstance(y_pred_proba, pd.DataFrame):
            res_df = pd.concat([res_df, y_pred_proba], axis=1)
        else:
            for i in range(y_pred_proba.shape[1]): res_df[f'prob_{i}'] = y_pred_proba[:, i]
        res_df.to_csv(os.path.join(results_folder, f'fold_{fold}_preds.csv'), index=False)
        
        # Plotting Confusion Matrix for the fold
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold}')
        pdf.savefig(); plt.close()

    pdf.close()
    
    # Summary of metrics across folds
    metrics_df = pd.DataFrame(all_metrics)
    num_metrics = metrics_df.select_dtypes(include=['number']).mean()
    num_metrics['fold'] = 'Average'
    pd.concat([metrics_df, pd.DataFrame([num_metrics])], ignore_index=True)\
      .to_csv(os.path.join(results_folder, 'all_metrics.csv'), index=False)
    
    # Plot and save average confusion matrix
    avg_cm = np.mean([m['confusion_matrix'] for m in all_metrics], axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Average Confusion Matrix')
    plt.savefig(os.path.join(results_folder, 'avg_cm.pdf')); plt.close()
    
    print(f"Done. Results saved to: {results_folder}")

if __name__ == "__main__":
    args = parse_args()
    setup_env(args)
    df = load_and_preprocess_data(args)
    run_kfold_training(args, df)