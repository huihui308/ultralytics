
import torch
from torch import nn


class SCAM_Attention(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(SCAM_Attention, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
 
    def forward(self, x):
        # x: [batch_size, num_channels, height, width]
        batch_size, _, height, width = x.size()
        
        # Step 1: Calculate the mean of x
        mean_x = torch.mean(x, dim=1, keepdim=True)
        
        # Step 2: Calculate the covariance matrix
        mean_x = mean_x.repeat(1, self.num_channels, 1, 1)
        x_centered = x - mean_x
        cov_xx = torch.mean(x_centered * x_centered, dim=1, keepdim=True)
        
        # Step 3: Calculate the inverse of covariance matrix
        cov_xx_inv = torch.inverse(cov_xx + torch.eye(height * width, device=x.device).repeat(batch_size, 1, 1) * self.epsilon)
        
        # Step 4: Calculate the spatial correlation matrix
        scam = cov_xx_inv
        
        return scam


if __name__ == '__main__':
    model = SCAM_Attention(num_channels=3)  # 假设输入是3通道图像
    input_tensor = torch.randn(1, 3, 10, 10)  # 假设输入是1个样本, 3通道, 高为10, 宽为10的图像
    scam = model(input_tensor)
    print(scam.shape)  # 输出应为 torch.Size([1, 1, 10, 10])