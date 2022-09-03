import torch
from torch import nn
import torch.nn.functional as F
from model.GCN_layers import GraphConvolution


class Model(nn.Module):
	def __init__(self, time_lag, pre_len, station_num, device):
		super().__init__()
		self.time_lag = time_lag
		self.pre_len = pre_len
		self.station_num = station_num
		self.device = device
		self.GCN_week = GraphConvolution(in_features=self.time_lag, out_features=self.time_lag).to(self.device)
		self.GCN_day = GraphConvolution(in_features=self.time_lag, out_features=self.time_lag).to(self.device)
		self.GCN_time = GraphConvolution(in_features=self.time_lag, out_features=self.time_lag).to(self.device)
		self.Conv2D = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1).to(self.device)
		self.linear1 = nn.Linear(in_features=8*self.time_lag*3*self.station_num, out_features=1024).to(self.device)
		self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
		self.linear3 = nn.Linear(in_features=512, out_features=self.station_num*self.pre_len).to(self.device)

	def forward(self, inflow, outflow, adj):
		inflow = inflow.to(self.device)
		outflow = outflow.to(self.device)
		adj = adj.to(self.device)
		# inflow = self.GCN(input=inflow, adj=adj)  # (64, 276, 10)
		inflow_week = inflow[:, :, 0:self.time_lag]
		inflow_day = inflow[:, :, self.time_lag:self.time_lag*2]
		inflow_time = inflow[:, :, self.time_lag*2:self.time_lag*3]
		inflow_week = self.GCN_week(x=inflow_week, adj=adj)  # (64, 276, 10)
		inflow_day = self.GCN_day(x=inflow_day, adj=adj)  # (64, 276, 10)
		inflow_time = self.GCN_time(x=inflow_time, adj=adj)  # (64, 276, 10)
		inflow = torch.cat([inflow_week, inflow_day, inflow_time], dim=2)
		output = inflow.unsqueeze(1)  # (64, 1, 276, 30)
		output = self.Conv2D(output)  # (64, 8, 276, 5)
		output = output.reshape(output.size()[0], -1)  # (64, 8*276*30)
		output = F.relu(self.linear1(output))  # (64, 1024)
		output = F.relu(self.linear2(output))  # (64, 512)
		output = self.linear3(output)  # (64, 276*pre_len)
		output = output.reshape(output.size()[0], self.station_num, self.pre_len)  # ( 64, 276, pre_len)
		return output
