"""
AEM正演模型预测与评估脚本
直接从 data/forward/ 加载数据，从 models/ 加载模型
每个模型的结果保存到 results/forward/<MODEL>/
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forward_model import AEMForwardModel
from utils import (
    get_device, calculate_metrics, print_metrics,
    plot_true_vs_predicted, plot_error_distribution,
    plot_error_distribution_comparison, plot_profile_comparison,
    analyze_by_height, analyze_by_layer, create_submission
)

DATA_DIR = 'data/forward'
SAVE_DIR = 'results/forward'
MODEL_TYPES = ['lstm', 'gru']


def evaluate(device):
    """在内部测试集上评估"""
    print("=" * 60)
    print("评估正演模型 (内部测试集)")
    print("=" * 60)

    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    scaler_y = joblib.load('models/scaler_y_forward.pkl')
    scaler_X = joblib.load('models/scaler_X_forward.pkl')
    heights_test = X_test[:, -1]

    all_results = {}

    for mtype in MODEL_TYPES:
        model_path = f'models/forward_{mtype}_best.pth'
        if not os.path.exists(model_path):
            print(f"  [!] Skip {mtype}")
            continue

        name = mtype.upper()
        model_dir = f'{SAVE_DIR}/{name}'
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n评估 {name}...")
        model = AEMForwardModel(model_type=mtype, input_dim=X_train.shape[1],
                                output_dim=y_test.shape[1], device=device)
        model.load(model_path)

        preds_scaled = model.predict(X_test)
        preds = scaler_y.inverse_transform(preds_scaled)
        targets = scaler_y.inverse_transform(y_test)

        metrics = calculate_metrics(targets, preds)
        print_metrics(metrics, f"Forward {name}")

        # 各模型独立图表
        plot_true_vs_predicted(targets, preds, model_name=f'Forward-{name}', n_samples=3,
                               save_path=f'{model_dir}/profile_overlay.png')
        plot_error_distribution(targets, preds, model_name=f'Forward-{name}',
                                save_path=f'{model_dir}/error_dist.png')
        analyze_by_layer(targets, preds, model_name=f'Forward-{name}',
                         save_path=f'{model_dir}/by_layer.png')

        # 按高度分析
        n_feat = scaler_X.n_features_in_
        dummy = np.zeros((len(heights_test), n_feat))
        dummy[:, -1] = heights_test
        heights_orig = scaler_X.inverse_transform(dummy)[:, -1]
        analyze_by_height(targets, preds, heights_orig, model_name=f'Forward-{name}',
                          save_path=f'{model_dir}/by_height.png')

        all_results[name] = (targets, preds, metrics)

    # 跨模型对比图（保存在顶层）
    if len(all_results) > 1:
        plot_error_distribution_comparison(all_results, save_path=f'{SAVE_DIR}/error_comparison.png')
        sample_idx = np.random.randint(0, len(list(all_results.values())[0][0]))
        plot_profile_comparison(all_results, sample_idx=sample_idx,
                                save_path=f'{SAVE_DIR}/profile_comparison.png')

    return all_results


def predict(device):
    """在待预测数据上预测"""
    print("\n" + "=" * 60)
    print("正演模型预测 (待预测数据)")
    print("=" * 60)

    X_predict = np.load(f'{DATA_DIR}/X_predict.npy')
    scaler_y = joblib.load('models/scaler_y_forward.pkl')
    print(f"  待预测: {X_predict.shape[0]} 条")

    for mtype in MODEL_TYPES:
        model_path = f'models/forward_{mtype}_best.pth'
        if not os.path.exists(model_path):
            continue
        name = mtype.upper()
        model_dir = f'{SAVE_DIR}/{name}'
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n使用 {name} 预测...")
        model = AEMForwardModel(model_type=mtype, input_dim=X_predict.shape[1],
                                output_dim=100, device=device)
        model.load(model_path)
        preds = scaler_y.inverse_transform(model.predict(X_predict))
        create_submission(preds, f'{DATA_DIR}/predict_source.csv',
                          output_path=f'{model_dir}/submission.csv', mode='forward')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()

    all_results = evaluate(device)

    if all_results:
        result_df = pd.DataFrame({k: v[2] for k, v in all_results.items()}).T
        result_df.index.name = 'Model'
        result_df.to_csv(f'{SAVE_DIR}/result.csv')
        print(f"\n评估指标已保存到 {SAVE_DIR}/result.csv")
        print(result_df.to_string())

    predict(device)
    print("\n正演评估完成!")


if __name__ == '__main__':
    main()
