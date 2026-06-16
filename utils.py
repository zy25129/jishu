"""
AEM通用工具模块
包含数据集、损失函数、评估指标、EMA、可视化、检查点管理
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
import os


# ============================================================
# 数据集与数据加载
# ============================================================

class AEMDataset(Dataset):
    """AEM PyTorch 数据集"""

    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """创建训练和验证数据加载器"""
    train_dataset = AEMDataset(X_train, y_train)
    val_dataset = AEMDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


# ============================================================
# EMA 指数移动平均
# ============================================================

class EMA:
    """指数移动平均：平滑模型参数，提升泛化能力"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ============================================================
# 组合损失函数
# ============================================================

class CombinedLoss(nn.Module):
    """MSE + 平滑约束损失"""

    def __init__(self, lambda_smooth=0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)
        smooth_loss = torch.mean((y_pred[:, 1:] - y_pred[:, :-1]) ** 2)
        return mse_loss + self.lambda_smooth * smooth_loss


# ============================================================
# 评估指标
# ============================================================

def calculate_metrics(y_true, y_pred):
    """计算 MSE, R2, RMSPE 评估指标
    注: RMSPE 使用归一化公式 RMSE/std(y_true)*100，适配对数尺度数据
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    std_true = np.std(y_true)
    rmspe = (rmse / (std_true + 1e-8)) * 100

    return {'MSE': mse, 'R2': r2, 'RMSPE': rmspe}


def print_metrics(metrics, prefix=''):
    """打印评估指标"""
    if prefix:
        print(f"\n{prefix} 评估指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


# ============================================================
# 设备
# ============================================================

def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


# ============================================================
# 检查点管理
# ============================================================

def save_checkpoint(model, optimizer, ema, epoch, path):
    """保存训练检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict() if ema else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, ema, path, device='cpu'):
    """加载训练检查点"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema and checkpoint.get('ema_state_dict'):
        ema.load_state_dict(checkpoint['ema_state_dict'])
    return checkpoint.get('epoch', 0)


# ============================================================
# EDA 可视化
# ============================================================

def plot_height_distribution(df, save_path=None):
    """绘制飞行高度分布直方图"""
    heights = df['Height'].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(heights, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(heights), color='r', linestyle='--', label=f'Mean={np.mean(heights):.2f}')
    ax.axvline(np.mean(heights) + np.std(heights), color='orange', linestyle=':', label=f'Std={np.std(heights):.2f}')
    ax.axvline(np.mean(heights) - np.std(heights), color='orange', linestyle=':')
    ax.set_xlabel('Height')
    ax.set_ylabel('Count')
    ax.set_title('飞行高度分布')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"高度分布图已保存: {save_path}")
    plt.close()


def plot_resistivity_heatmap(df, save_path=None, n_samples=200):
    """绘制电阻率热力图"""
    res_cols = [f'resLayer{i}' for i in range(1, 101)]
    data = df[res_cols].values[:n_samples]
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Sample Index')
    ax.set_title('电阻率热力图')
    plt.colorbar(im, ax=ax, label='Resistivity')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"电阻率热力图已保存: {save_path}")
    plt.close()


def plot_response_mean_std(df, save_path=None):
    """绘制响应均值+/-标准差带图"""
    R_cols = [f'R{i}' for i in range(1, 101)]
    data = df[R_cols].values
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    channels = np.arange(1, 101)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(channels, mean_vals, 'b-', linewidth=1.5, label='Mean')
    ax.fill_between(channels, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, color='blue', label='+/-1 Std')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Response Value')
    ax.set_title('电磁响应均值±标准差带图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"响应均值-标准差带图已保存: {save_path}")
    plt.close()


# ============================================================
# 对比可视化
# ============================================================

def plot_training_history(histories, save_path=None):
    """绘制多模型训练历史对比图"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (name, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], '-', color=color, alpha=0.7, label=f'{name} Train')
        axes[0].plot(epochs, hist['val_loss'], '--', color=color, label=f'{name} Val')
        axes[1].plot(epochs, hist['val_r2'], '-', color=color, label=name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练和验证损失曲线')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2')
    axes[1].set_title('验证集R²曲线')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史对比图已保存: {save_path}")
    plt.close()


def plot_true_vs_predicted(y_true, y_pred, model_name='', n_samples=3, save_path=None):
    """绘制真实值 vs 预测值剖面叠合图"""
    n_total = len(y_true)
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    layers = np.arange(1, 101)

    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 5))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(layers, y_true[idx], 'b-', linewidth=1.5, label='True')
        ax.plot(layers, y_pred[idx], 'r--', linewidth=1.5, label='Predicted')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Resistivity')
        ax.set_title(f'样本 {idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'真实值 vs 预测值剖面叠合图 — {model_name}', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"剖面叠合图已保存: {save_path}")
    plt.close()


def plot_error_distribution(y_true, y_pred, model_name='', save_path=None):
    """绘制误差分布直方图"""
    error = (y_true - y_pred).flatten()
    mean_err = np.mean(error)
    std_err = np.std(error)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(error, bins=80, density=True, alpha=0.7, edgecolor='black')

    x_range = np.linspace(mean_err - 4 * std_err, mean_err + 4 * std_err, 200)
    ax.plot(x_range, stats.norm.pdf(x_range, mean_err, std_err), 'r-', linewidth=2, label='Normal Fit')

    ax.axvline(mean_err, color='orange', linestyle='--', label=f'Mean={mean_err:.4f}')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title(f'误差分布 — {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"误差分布图已保存: {save_path}")
    plt.close()


def plot_error_distribution_comparison(all_results, save_path=None):
    """绘制多模型误差分布对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (name, (y_true, y_pred, _)), color in zip(all_results.items(), colors):
        error = (y_true - y_pred).flatten()
        ax.hist(error, bins=80, density=True, alpha=0.4, color=color, label=name, edgecolor='none')

    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('误差分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"误差分布对比图已保存: {save_path}")
    plt.close()


def plot_profile_comparison(all_results, sample_idx=0, save_path=None):
    """绘制多模型在同一样本上的剖面叠合对比"""
    layers = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(10, 6))

    y_true_ref = None
    for name, (y_true, y_pred, _) in all_results.items():
        if y_true_ref is None:
            ax.plot(layers, y_true[sample_idx], 'k-', linewidth=2, label='True')
            y_true_ref = True
        ax.plot(layers, y_pred[sample_idx], '--', linewidth=1.5, label=name)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Resistivity')
    ax.set_title(f'剖面叠合对比 — 样本 {sample_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"剖面叠合对比图已保存: {save_path}")
    plt.close()


# ============================================================
# 按高度 / 按深度层分析
# ============================================================

def analyze_by_height(y_true, y_pred, heights, model_name='', n_bins=5, save_path=None):
    """
    按高度区间分组计算 MSE / R2 / RMSPE
    heights: 原始高度值数组 (未标准化), shape=(n_samples,)
    """
    results = []
    bin_edges = np.linspace(heights.min(), heights.max(), n_bins + 1)
    for i in range(n_bins):
        mask = (heights >= bin_edges[i]) & (heights < bin_edges[i + 1])
        if mask.sum() < 2:
            continue
        m = calculate_metrics(y_true[mask], y_pred[mask])
        m['height_range'] = f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}'
        m['n_samples'] = int(mask.sum())
        results.append(m)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index('height_range')
    print(f"\n{model_name} 按高度分组指标:")
    print(df.to_string())

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric in zip(axes, ['MSE', 'R2', 'RMSPE']):
            ax.bar(range(len(df)), df[metric], tick_label=df.index, alpha=0.7)
            ax.set_xlabel('Height Range')
            ax.set_ylabel(metric)
            ax.set_title(f'{model_name} - 按高度 {metric}')
            ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"按高度分析图已保存: {save_path}")
        plt.close()

    return df


def analyze_by_layer(y_true, y_pred, model_name='', save_path=None):
    """
    逐深度层计算 MSE 和 R2
    y_true, y_pred: shape=(n_samples, 100)
    """
    n_layers = y_true.shape[1]
    layer_mse = np.array([np.mean((y_true[:, i] - y_pred[:, i]) ** 2) for i in range(n_layers)])
    layer_r2 = []
    for i in range(n_layers):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        layer_r2.append(1 - ss_res / (ss_tot + 1e-8))
    layer_r2 = np.array(layer_r2)

    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(range(1, n_layers + 1), layer_mse, 'b-', linewidth=1.2)
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('MSE')
        axes[0].set_title(f'{model_name} - 逐层MSE')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(1, n_layers + 1), layer_r2, 'r-', linewidth=1.2)
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('R2')
        axes[1].set_title(f'{model_name} - 逐层R²')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"按层分析图已保存: {save_path}")
        plt.close()

    return layer_mse, layer_r2


# ============================================================
# 提交文件生成
# ============================================================

def create_submission(predictions, source_csv_path, output_path='submission.csv', mode='inverse'):
    """创建提交文件
    source_csv_path: predict_source.csv 路径
    mode: 'inverse'=输出resLayer列, 'forward'=输出R列
    """
    source_df = pd.read_csv(source_csv_path)
    submission = source_df[['yindex', 'Xindex', 'Height']].copy()
    if mode == 'inverse':
        columns = [f'resLayer{i}' for i in range(1, 101)]
    else:
        columns = [f'R{i}' for i in range(1, 101)]
    pred_df = pd.DataFrame(predictions, columns=columns, index=submission.index)
    submission = pd.concat([submission, pred_df], axis=1)
    submission.to_csv(output_path, index=False)
    print(f"  已保存: {output_path} ({submission.shape[0]} 条)")
    return submission
