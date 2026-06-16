"""
AEM反演模型模块
- InverseLSTMEncoderDecoder: LSTM编码器-解码器 + 多头注意力（主模型）
- InverseCNN: 1D卷积对比模型
- InverseMLP: 多层感知机对比模型
- InverseTCN: 时间卷积网络对比模型
"""
import numpy as np
import torch
import torch.nn as nn
from utils import CombinedLoss, EMA, save_checkpoint, load_checkpoint


# ============================================================
# LSTM 编码器-解码器 + 多头注意力（主模型）
# ============================================================

class InverseLSTMEncoderDecoder(nn.Module):
    """
    反演主模型：LSTM编码器-解码器 + 多头注意力
    输入: 101维 (100道电磁响应 + 飞行高度)
    输出: 100维 (100层电阻率)
    """

    def __init__(self, input_dim=101, output_dim=100, hidden_dim=128,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 输入投影
        self.input_proj = nn.Linear(1, hidden_dim)

        # 编码器
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层: 将每个时间步的 hidden_dim 压缩到 1，再投影到 output_dim
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # 当输入时间步数 != output_dim 时，用投影对齐
        self.output_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        # 输入投影: (batch, 101) -> (batch, 101, 1) -> (batch, 101, hidden_dim)
        x_seq = x.unsqueeze(-1)
        x_proj = self.input_proj(x_seq)

        # 编码
        encoder_out, _ = self.encoder(x_proj)

        # 多头自注意力 + 残差
        attn_out, _ = self.multihead_attn(encoder_out, encoder_out, encoder_out)
        encoder_out = self.attn_norm(encoder_out + attn_out)

        # 解码
        decoder_out, _ = self.decoder(encoder_out)

        # 残差连接
        decoder_out = decoder_out + encoder_out

        output = self.fc_out(decoder_out).squeeze(-1)  # (batch, 101)
        if self.output_proj is not None:
            output = self.output_proj(output)  # (batch, 100)
        return output


# ============================================================
# CNN 对比模型
# ============================================================

class InverseCNN(nn.Module):
    """
    反演对比模型：1D卷积网络
    """

    def __init__(self, input_dim=101, output_dim=100, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # input_dim -> floor(input_dim/2) -> floor/2 -> floor/2
        conv_out_dim = input_dim // 8  # 101//8 = 12
        self.classifier = nn.Sequential(
            nn.Linear(256 * conv_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 101)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


# ============================================================
# MLP 对比模型
# ============================================================

class InverseMLP(nn.Module):
    """
    反演对比模型：多层感知机
    """

    def __init__(self, input_dim=101, output_dim=100, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# TCN 对比模型
# ============================================================

class TCNBlock(nn.Module):
    """时间卷积网络基本块"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 因果裁剪：去掉右侧填充
        self.chomp_size = padding

        # 残差投影（通道数不同时）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.chomp_size > 0:
            out = out[:, :, :-self.chomp_size]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.chomp_size > 0:
            out = out[:, :, :-self.chomp_size]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class InverseTCN(nn.Module):
    """
    反演对比模型：时间卷积网络
    使用膨胀因果卷积，感受野覆盖全部输入通道
    """

    def __init__(self, input_dim=101, output_dim=100, num_channels=64,
                 kernel_size=3, dropout=0.1):
        super().__init__()

        dilations = [1, 2, 4, 8]
        layers = []
        in_ch = 1
        for d in dilations:
            layers.append(TCNBlock(in_ch, num_channels, kernel_size, d, dropout))
            in_ch = num_channels

        self.tcn = nn.Sequential(*layers)
        self.fc_out = nn.Sequential(
            nn.Linear(num_channels * input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 100)
        x = self.tcn(x)  # (batch, num_channels, 100)
        x = x.reshape(x.size(0), -1)  # (batch, num_channels*100)
        output = self.fc_out(x)  # (batch, 100)
        return output


# ============================================================
# 反演模型包装器
# ============================================================

class AEMInverseModel:
    """反演模型统一包装器：支持 lstm/cnn/mlp/tcn"""

    def __init__(self, model_type='lstm', input_dim=101, output_dim=100,
                 lr=0.001, device='cpu', lambda_smooth=0.01,
                 total_epochs=100, warmup_epochs=10):
        self.device = device
        self.model_type = model_type

        # 实例化网络
        if model_type == 'lstm':
            self.model = InverseLSTMEncoderDecoder(input_dim, output_dim).to(device)
        elif model_type == 'cnn':
            self.model = InverseCNN(input_dim, output_dim).to(device)
        elif model_type == 'mlp':
            self.model = InverseMLP(input_dim, output_dim).to(device)
        elif model_type == 'tcn':
            self.model = InverseTCN(input_dim, output_dim).to(device)
        else:
            raise ValueError(f"未知反演模型类型: {model_type}")

        # 损失函数
        self.criterion = CombinedLoss(lambda_smooth)

        # 优化器: AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # 调度器: 线性预热 + 余弦退火
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # EMA
        self.ema = EMA(self.model, decay=0.999)

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.ema.update()

            total_loss += loss.item() * len(X_batch)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        """验证"""
        self.ema.apply_shadow()
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * len(X_batch)
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        self.ema.restore()

        avg_loss = total_loss / len(val_loader.dataset)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        return avg_loss, all_preds, all_targets

    def predict(self, X):
        """预测"""
        self.ema.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_tensor).cpu().numpy()
        self.ema.restore()
        return preds

    def save(self, path):
        """保存检查点"""
        save_checkpoint(self.model, self.optimizer, self.ema, 0, path)

    def load(self, path):
        """加载检查点"""
        load_checkpoint(self.model, self.optimizer, self.ema, path, self.device)


# 测试
if __name__ == '__main__':
    device = torch.device('cpu')
    for mtype in ['lstm', 'cnn', 'mlp', 'tcn']:
        model = AEMInverseModel(model_type=mtype, device=device)
        x = torch.randn(4, 101)
        out = model.model(x)
        print(f"Inverse {mtype}: input {x.shape} -> output {out.shape}")
