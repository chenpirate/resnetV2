'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-05 18:38:15
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-05 20:25:03
FilePath: /resnetV2/utils/tensor_to_prob_dist.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch.distributions import Normal, StudentT

def map_tensor_to_dist(input_tensor, dist_type="gaussian"):
  # Check the shape of the input tensor
  shape = input_tensor.shape
  if len(shape) < 2:
    raise ValueError("Input tensor must have at least two dimensions")
  # Check the distribution type
  if dist_type not in ["gaussian", "t"]:
    raise ValueError("Distribution type must be either 'gaussian' or 't'")
  # Compute the mean and standard deviation of the input tensor along the last axis
  y = None
  # Normalize the input tensor by subtracting the mean and dividing by the standard deviation
  # normalized_tensor = (input_tensor - mean) / std
  # Compute the cumulative distribution function of the normalized tensor according to the distribution type
  if dist_type == "gaussian":
    # Use the normal distribution formula
    mean = torch.mean(input_tensor, dim=[0, 1], keepdim=False)
    std = torch.std(input_tensor, dim=[0, 1], keepdim=False)
    normal = Normal(loc=mean, scale=std) # 创建高斯分布对象
    y = normal.log_prob(input_tensor).sum(dim=-1)
    y = torch.softmax(y, dim=-1)
  else:
    # Use the t distribution formula with degrees of freedom equal to the size of the last axis minus one
    student = StudentT(df=shape[0])
    y = student.log_prob(input_tensor).sum(dim=-1)
    y = torch.softmax(y, dim=-1)
  # Return the probability value tensor
  return y


if __name__ == '__main__':
    test1_input = torch.rand(2048, 4)
#   print(f"test_input:{test_input}")
    out1 = map_tensor_to_dist(test1_input, "gaussian")
    print(f"out1:{out1.size()}")
  # import torch
  # import torch.distributions as dist

  # # 生成一个数据集，服从均值为[2, 3]，协方差矩阵为[[1, 0.5], [0.5, 2]]的二元高斯分布
  # data = torch.randn(100, 2) * torch.squeeze(torch.tensor([[1, 0.5], [0.5, 2]])) + torch.squeeze(torch.tensor([2, 3]))

  # # 计算数据集的均值和协方差矩阵
  # loc = torch.mean(data, dim=0)
  # covariance_matrix = torch.cov(data.T)

  # # 创建一个多元高斯分布的实例
  # multivariate_normal = dist.MultivariateNormal(loc, covariance_matrix)

  # # 计算一个新的数据点[2048, 4]的对数概率密度函数
  # log_prob = multivariate_normal.log_prob(torch.tensor([2048, 4])) # 这里修改了数据点的值
  # print(log_prob) # tensor(-2076.9778)
  # import torch
  # x1 = torch.rand(2048, 4)
  # x2 = torch.rand(2048, 2)

  # # 假设 x1 是形状为 (2048,4) 的张量，x2 是形状为 (2048,2) 的张量
  # # x1 = torch.softmax(x1, dim=1) # 对 x1 的每一行进行 softmax 操作
  # # x2 = torch.softmax(x2, dim=1) # 对 x2 的每一行进行 softmax 操作

  # from torch.distributions import Normal, StudentT
  # # 假设我们想要映射到均值为 0，标准差为 1 的高斯分布
  # normal = Normal(loc=0, scale=1) # 创建高斯分布对象
  # # 假设我们想要映射到自由度为 2 的 t 分布
  # student = StudentT(df=2048) # 创建 t 分布对象
  # # 对 x1 的每一行进行高斯分布的映射
  # y1 = normal.log_prob(x1).sum(dim=1) # 计算每一行的对数概率密度值，并求和得到一个形状为 (2048,) 的张量
  # # 对 x2 的每一行进行 t 分布的映射
  # y2 = student.log_prob(x2).sum(dim=1) # 计算每一行的对数概率密度值，并求和得到一个形状为 (2048,) 的张量

  # y1 = torch.softmax(y1, dim=-1) # 对 x1 的每一行进行 softmax 操作
  # y2 = torch.softmax(y2, dim=-1)

  # from torch.nn.functional import kl_div
  # # 假设 y1 是高斯分布的概率密度函数，y2 是 t 分布的概率密度函数
  # kl = kl_div(y1.log(), y2, reduction='batchmean') # 计算两个张量之间的 KL 散度，并取平均值
  # print(f"x1:{x1.size()} x2:{x2.size()} y1:{y1.size()} y2:{y2.size()}")
  # print(f"kl:{kl}")




