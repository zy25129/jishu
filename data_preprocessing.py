"""
AEM数据预处理脚本
运行一次，将所有处理后的数据保存到 data/ 目录
后续训练和预测脚本直接从 data/ 加载，无需重复预处理
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

from utils import (
    plot_height_distribution, plot_resistivity_heatmap,
    plot_response_mean_std
)

DATA_DIR = Path('D:/Data/大学课程数据/项目技术实训1/AEM数据及资料')
SAVE_DIR = Path('data')


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(SAVE_DIR / 'forward', exist_ok=True)
    os.makedirs(SAVE_DIR / 'inverse', exist_ok=True)

    # ============================================================
    # 1. 加载原始数据
    # ============================================================
    print("加载数据...")
    sample_data = pd.read_csv(DATA_DIR / 'train.csv')
    predict_data = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"样本数据 (train.csv): {sample_data.shape[0]} 条")
    print(f"待预测数据 (test.csv): {predict_data.shape[0]} 条")

    # 缺失值/无穷值检查
    nan_cnt = sample_data.isnull().sum().sum()
    inf_cnt = np.isinf(sample_data.select_dtypes(include=[np.number]).values).sum()
    print(f"  样本数据: 缺失值={nan_cnt}, 无穷值={inf_cnt}")

    # ============================================================
    # 2. 拆分待预测数据为正演/反演两组
    # ============================================================
    res_cols = [f'resLayer{i}' for i in range(1, 101)]
    R_cols = [f'R{i}' for i in range(1, 101)]
    is_sentinel = (predict_data[res_cols].values >= 9998).all(axis=1)

    forward_predict_df = predict_data[~is_sentinel].reset_index(drop=True)
    inverse_predict_df = predict_data[is_sentinel].reset_index(drop=True)

    print(f"  正演待预测: {len(forward_predict_df)} 条 (resLayer为真实值)")
    print(f"  反演待预测: {len(inverse_predict_df)} 条 (resLayer为9999占位)")

    # ============================================================
    # 3. EDA
    # ============================================================
    print("\n运行探索性分析...")
    plot_height_distribution(sample_data, 'results/eda_height_distribution.png')
    plot_resistivity_heatmap(sample_data, 'results/eda_resistivity_heatmap.png')
    plot_response_mean_std(sample_data, 'results/eda_response_mean_std.png')

    # 统计信息
    print("\n=== 样本数据统计 (8000条) ===")
    print(f"电阻率: [{sample_data[res_cols].min().min():.2f}, {sample_data[res_cols].max().max():.2f}], "
          f"均值={sample_data[res_cols].mean().mean():.2f}")
    print(f"响应值: [{sample_data[R_cols].min().min():.4f}, {sample_data[R_cols].max().max():.4f}], "
          f"均值={sample_data[R_cols].mean().mean():.4f}")
    print(f"高度: [{sample_data['Height'].min():.2f}, {sample_data['Height'].max():.2f}]")

    # ============================================================
    # 4. 正演数据: resLayer+Height(101) -> R(100)
    # ============================================================
    print("\n准备正演数据...")
    X_fwd = sample_data[res_cols + ['Height']].values
    y_fwd = sample_data[R_cols].values

    scaler_X_fwd = StandardScaler()
    scaler_y_fwd = StandardScaler()
    X_fwd_scaled = scaler_X_fwd.fit_transform(X_fwd)
    y_fwd_scaled = scaler_y_fwd.fit_transform(y_fwd)

    X_tv, X_te, y_tv, y_te = train_test_split(X_fwd_scaled, y_fwd_scaled, test_size=0.15, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.15/0.85, random_state=42)

    # 保存正演数据
    fwd_dir = SAVE_DIR / 'forward'
    np.save(fwd_dir / 'X_train.npy', X_tr)
    np.save(fwd_dir / 'X_val.npy', X_va)
    np.save(fwd_dir / 'X_test.npy', X_te)
    np.save(fwd_dir / 'y_train.npy', y_tr)
    np.save(fwd_dir / 'y_val.npy', y_va)
    np.save(fwd_dir / 'y_test.npy', y_te)

    # 正演待预测数据
    X_fwd_pred = forward_predict_df[res_cols + ['Height']].values
    X_fwd_pred_scaled = scaler_X_fwd.transform(X_fwd_pred)
    np.save(fwd_dir / 'X_predict.npy', X_fwd_pred_scaled)
    forward_predict_df.to_csv(fwd_dir / 'predict_source.csv', index=False)

    joblib.dump(scaler_X_fwd, 'models/scaler_X_forward.pkl')
    joblib.dump(scaler_y_fwd, 'models/scaler_y_forward.pkl')

    print(f"  训练集: {X_tr.shape[0]}, 验证集: {X_va.shape[0]}, 测试集: {X_te.shape[0]}")
    print(f"  待预测: {X_fwd_pred_scaled.shape[0]} 条")
    print(f"  已保存到 {fwd_dir}/")

    # ============================================================
    # 5. 反演数据: R+Height(101) -> resLayer(100)
    # ============================================================
    print("\n准备反演数据...")
    X_inv = sample_data[R_cols + ['Height']].values
    y_inv = sample_data[res_cols].values

    scaler_X_inv = StandardScaler()
    scaler_y_inv = StandardScaler()
    X_inv_scaled = scaler_X_inv.fit_transform(X_inv)
    y_inv_scaled = scaler_y_inv.fit_transform(y_inv)

    X_tv, X_te, y_tv, y_te = train_test_split(X_inv_scaled, y_inv_scaled, test_size=0.15, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.15/0.85, random_state=42)

    # 保存反演数据
    inv_dir = SAVE_DIR / 'inverse'
    np.save(inv_dir / 'X_train.npy', X_tr)
    np.save(inv_dir / 'X_val.npy', X_va)
    np.save(inv_dir / 'X_test.npy', X_te)
    np.save(inv_dir / 'y_train.npy', y_tr)
    np.save(inv_dir / 'y_val.npy', y_va)
    np.save(inv_dir / 'y_test.npy', y_te)

    # 反演待预测数据
    X_inv_pred = inverse_predict_df[R_cols + ['Height']].values
    X_inv_pred_scaled = scaler_X_inv.transform(X_inv_pred)
    np.save(inv_dir / 'X_predict.npy', X_inv_pred_scaled)
    inverse_predict_df.to_csv(inv_dir / 'predict_source.csv', index=False)

    joblib.dump(scaler_X_inv, 'models/scaler_X_inverse.pkl')
    joblib.dump(scaler_y_inv, 'models/scaler_y_inverse.pkl')

    print(f"  训练集: {X_tr.shape[0]}, 验证集: {X_va.shape[0]}, 测试集: {X_te.shape[0]}")
    print(f"  待预测: {X_inv_pred_scaled.shape[0]} 条")
    print(f"  已保存到 {inv_dir}/")

    # ============================================================
    # 6. 完成
    # ============================================================
    print("\n" + "=" * 50)
    print("数据预处理完成! 所有文件已保存到 data/ 和 models/")
    print(f"  data/forward/  (正演: 训练/验证/测试/待预测)")
    print(f"  data/inverse/  (反演: 训练/验证/测试/待预测)")
    print(f"  models/        (标准化器)")
    print(f"  results/       (EDA图表)")
    print("=" * 50)


if __name__ == '__main__':
    main()
