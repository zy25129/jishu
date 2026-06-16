"""
AEM反演模型训练脚本
直接从 data/inverse/ 加载预处理好的数据
结果保存到 results/inverse/
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inverse_model import AEMInverseModel
from utils import (
    create_dataloaders, calculate_metrics, print_metrics,
    plot_training_history, get_device
)

DATA_DIR = 'data/inverse'
SAVE_DIR = 'results/inverse'


def train_single_model(model_type, X_train, y_train, X_val, y_val,
                        epochs, batch_size, lr, device, patience,
                        lambda_smooth, warmup_epochs):
    print(f"\n{'='*60}")
    print(f"训练反演模型: {model_type.upper()}")
    print(f"{'='*60}")

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    model = AEMInverseModel(
        model_type=model_type, input_dim=X_train.shape[1],
        output_dim=y_train.shape[1], lr=lr, device=device,
        lambda_smooth=lambda_smooth, total_epochs=epochs, warmup_epochs=warmup_epochs)

    model_path = f'models/inverse_{model_type}_best.pth'
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("开始训练...")
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader)
        val_loss, val_preds, val_targets = model.validate(val_loader)
        val_metrics = calculate_metrics(val_targets, val_preds)
        model.scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_metrics['R2'])

        if (epoch + 1) % 10 == 0:
            lr_now = model.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}]  LR: {lr_now:.6f}")
            print(f"  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
            print(f"  Val R2: {val_metrics['R2']:.6f}  Val RMSPE: {val_metrics['RMSPE']:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(model_path)
            print(f"  [OK] Saved best (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [!] Early stopping at epoch {epoch+1}")
                break

    model.load(model_path)
    _, final_preds, final_targets = model.validate(val_loader)
    final_metrics = calculate_metrics(final_targets, final_preds)
    print_metrics(final_metrics, f"Inverse {model_type.upper()}")
    return history, final_metrics


def main():
    parser = argparse.ArgumentParser(description='AEM反演模型训练')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lambda_smooth', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()

    print(f"从 {DATA_DIR}/ 加载数据...")
    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    X_val = np.load(f'{DATA_DIR}/X_val.npy')
    y_train = np.load(f'{DATA_DIR}/y_train.npy')
    y_val = np.load(f'{DATA_DIR}/y_val.npy')
    print(f"  训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")

    configs = [('lstm', 'LSTM编码器-解码器+多头注意力 (主模型)'),
               ('cnn', '1D CNN对比模型'),
               ('mlp', 'MLP对比模型'),
               ('tcn', 'TCN时间卷积网络对比模型')]

    all_histories, all_metrics = {}, {}
    for mtype, desc in configs:
        print(f"\n>>> {desc}")
        h, m = train_single_model(
            mtype, X_train, y_train, X_val, y_val,
            args.epochs, args.batch_size, args.lr, device,
            args.patience, args.lambda_smooth, args.warmup_epochs)
        all_histories[f'{mtype.upper()}'] = h
        all_metrics[f'{mtype.upper()}'] = m

    result_df = pd.DataFrame(all_metrics).T
    result_df.index.name = 'Model'
    result_df.to_csv(f'{SAVE_DIR}/result.csv')
    print(f"\n评估指标已保存到 {SAVE_DIR}/result.csv")
    print(result_df.to_string())

    plot_training_history(all_histories, f'{SAVE_DIR}/training_history.png')
    print("\n反演模型训练完成!")


if __name__ == '__main__':
    main()
