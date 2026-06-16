"""
AEM正演模型模块
- ForwardLSTMEncoderDecoder: LSTM编码器-解码器 + 注意力 + 残差（主模型）
- ForwardGRU: GRU对比模型
"""
import numpy as np
import torch
import torch.nn as nn
from utils import CombinedLoss, EMA, save_checkpoint, load_checkpoint


# ============================================================
# LSTM 编码器-解码器 + 注意力 + 残差（主模型）
# ============================================================

class ForwardLSTMEncoderDecoder(nn.Module):
    """
    正演主模型：LSTM编码器-解码器 + 注意力 + 残差连接
    输入: 101维 (100层电阻率 + 飞行高度)
    输出: 100维 (100道电磁响应)
    """

    def __init__(self, input_dim=101, output_dim=100, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 输入投影：将每个特征映射到hidden_dim维
        self.input_proj = nn.Linear(1, hidden_dim)

        # 编码器LSTM
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention_W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_V = nn.Linear(hidden_dim, 1)

        # 残差连接投影（双向 → 单向）
        self.residual_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # 解码器：用context初始化隐状态，输入零向量
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder_input_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出层：从decoder最后时间步映射到输出维度
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 输入投影: (batch, 101) → (batch, 101, 1) → (batch, 101, hidden_dim)
        x_seq = x.unsqueeze(-1)
        x_proj = self.input_proj(x_seq)

        # 编码
        encoder_out, _ = self.encoder(x_proj)  # (batch, 101, hidden_dim*2)

        # 注意力
        attn_scores = self.attention_V(torch.tanh(self.attention_W(encoder_out)))
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * encoder_out, dim=1)  # (batch, hidden_dim*2)

        # 残差连接
        context_proj = self.residual_proj(context)  # (batch, hidden_dim)

        # 解码：用context作为初始输入，单步解码
        decoder_input = self.decoder_input_proj(context_proj).unsqueeze(1)  # (batch, 1, hidden_dim)
        decoder_out, _ = self.decoder(decoder_input)  # (batch, 1, hidden_dim)
        decoder_out = decoder_out.squeeze(1) + context_proj  # 残差 (batch, hidden_dim)

        output = self.fc_out(decoder_out)  # (batch, 100)
        return output


# ============================================================
# GRU 对比模型
# ============================================================

class ForwardGRU(nn.Module):
    """
    正演对比模型：GRU编码器
    结构简单，用于与LSTM模型对比
    """

    def __init__(self, input_dim=101, output_dim=100, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(1, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x_seq = x.unsqueeze(-1)
        x_proj = self.input_proj(x_seq)
        gru_out, _ = self.gru(x_proj)

        # 取最后一个时间步
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim*2)

        # 重复到100步输出
        output = self.fc_out(last_hidden)  # (batch, output_dim)
        return output


# ============================================================
# 正演模型包装器
# ============================================================

class AEMForwardModel:
    """正演模型统一包装器：训练、验证、预测、保存/加载"""

    def __init__(self, model_type='lstm', input_dim=101, output_dim=100,
                 lr=0.001, device='cpu', lambda_smooth=0.01,
                 total_epochs=100, warmup_epochs=10):
        self.device = device
        self.model_type = model_type

        # 实例化网络
        if model_type == 'lstm':
            self.model = ForwardLSTMEncoderDecoder(input_dim, output_dim).to(device)
        elif model_type == 'gru':
            self.model = ForwardGRU(input_dim, output_dim).to(device)
        else:
            raise ValueError(f"未知正演模型类型: {model_type}")

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
    for mtype in ['lstm', 'gru']:
        model = AEMForwardModel(model_type=mtype, device=device)
        x = torch.randn(4, 101)
        out = model.model(x)
        print(f"Forward {mtype}: input {x.shape} -> output {out.shape}")
