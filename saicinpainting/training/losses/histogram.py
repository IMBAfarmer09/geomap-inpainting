import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramLoss(nn.Module):
    def __init__(self, n_bins=256, weight=5.0, min_val=0.0, max_val=1.0):
        """
        n_bins: 直方图的 bin 数
        min_val, max_val: 像素值范围（通常 [0,1] 或 [-1,1]）
        """
        super(HistogramLoss, self).__init__()
        self.n_bins = n_bins
        self.weight = weight
        self.min_val = min_val
        self.max_val = max_val
        # 计算每个 bin 的宽度
        self.step = (max_val - min_val) / float(n_bins - 1)
        # 预先生成所有 bin 中心点 t_i，register_buffer 保证随模型一起移动到 GPU/CPU
        t = torch.linspace(min_val, max_val, n_bins).view(n_bins, 1)
        self.register_buffer('t', t)

    def forward(self, x, y):
        """
        x, y: 二者形状都是 (B, C, H, W)，像素值应在 [min_val, max_val] 区间
        返回值: 一个标量，表示两幅图颜色直方图的 MSE 损失
        """
        # 1) 将 batch、通道都展平成一个长向量：所有像素一起做直方图
        x_flat = x.contiguous().view(-1)   # (B*C*H*W,)
        y_flat = y.contiguous().view(-1)   # (B*C*H*W,)
        N_x = x_flat.numel()
        N_y = y_flat.numel()

        # 2) 对每个像素值 s 贡献到相邻两个 bin 的权重：
        #    weight_i(s) = max(0, 1 - |s - t_i| / step)
        #    这样每个样本只会对离它最近的两个 bin 产生线性插值
        #    总体 histogram_i = sum_s weight_i(s) / N

        # x 映射
        x_repeat = x_flat.unsqueeze(0).expand(self.n_bins, -1)     # (n_bins, N_x)
        diff_x   = torch.abs(x_repeat - self.t)                     # (n_bins, N_x)
        weights_x = F.relu(1.0 - diff_x / self.step)                # (n_bins, N_x)
        hist_x = weights_x.sum(dim=1) / float(N_x)                  # (n_bins,)

        # y 映射（同理）
        y_repeat = y_flat.unsqueeze(0).expand(self.n_bins, -1)     # (n_bins, N_y)
        diff_y   = torch.abs(y_repeat - self.t)                     # (n_bins, N_y)
        weights_y = F.relu(1.0 - diff_y / self.step)                # (n_bins, N_y)
        hist_y = weights_y.sum(dim=1) / float(N_y)                  # (n_bins,)

        # 3) 计算两直方图的均方误差
        loss = F.mse_loss(hist_x, hist_y)
        return loss


if __name__ == '__main__':
    import torch

    # 假设上面已经定义了 HistogramLoss
    loss_fn = HistogramLoss(n_bins=4, min_val=0.0, max_val=1.0)

    # 测试 1：完全相同的“图像”，直方图一致，损失应为 0
    x = torch.tensor([[[[0.0, 1.0],
                        [0.0, 1.0]]]])  # shape (1,1,2,2)
    y = x.clone()
    loss_same = loss_fn(x, y).item()
    print(f"Identical inputs loss: {loss_same:.6f}  (expected ~0)")

    # 测试 2：值互换后的“图像”，分布不同，损失应 > 0
    y_swapped = torch.tensor([[[[1.0, 0.0],
                                [1.0, 0.0]]]])
    loss_diff = loss_fn(x, y_swapped).item()
    print(f"Swapped inputs loss:   {loss_diff:.6f}  (expected >0)")

    # 测试 3：随机噪声 vs. 常量图像
    x_rand = torch.rand(1, 1, 4, 4)
    y_const = torch.zeros(1, 1, 4, 4)
    loss_rand = loss_fn(x_rand, y_const).item()
    print(f"Random vs constant loss: {loss_rand:.6f}  (expected >>0)")
