import numpy as np
import os, time, torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import GetLaplacian
from model.main_model import Model
import matplotlib.pyplot as plt
from utils.metrics import Metrics, Metrics_1d
from data.get_dataloader import get_inflow_dataloader, get_outflow_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epoch_num = 1000
lr = 0.001
time_interval = 15
time_lag = 10
tg_in_one_day = 72
forecast_day_number = 5
pre_len = 1
batch_size = 32
station_num = 276
model_type = 'ours'
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow = \
	get_inflow_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
outflow_data_loader_train, outflow_data_loader_val, outflow_data_loader_test, max_outflow, min_outflow = \
	get_outflow_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)

# get normalized adj
adjacency = np.loadtxt('./data/adjacency.csv', delimiter=",")
adjacency = torch.tensor(GetLaplacian(adjacency).get_normalized_adj(station_num)).type(torch.float32).to(device)

global_start_time = time.time()
writer = SummaryWriter()

model = Model(time_lag, pre_len, station_num, device)

if torch.cuda.is_available():
	model.cuda()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

path = 'D:/subway flow prediction_for book/save_model/1_ours2021_04_12_14_34_43/model_dict_checkpoint_29_0.00002704.pth'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint, strict=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# test
result = []
result_original = []
if not os.path.exists('result/prediction'):
	os.makedirs('result/prediction/')
if not os.path.exists('result/original'):
	os.makedirs('result/original')
with torch.no_grad():
	model.eval()
	test_loss = 0
	for inflow_te, outflow_te in zip(enumerate(inflow_data_loader_test), enumerate(outflow_data_loader_test)):
		i_batch, (test_inflow_X, test_inflow_Y, test_inflow_Y_original) = inflow_te
		i_batch, (test_outflow_X, test_outflow_Y, test_outflow_Y_original) = outflow_te
		test_inflow_X, test_inflow_Y = test_inflow_X.type(torch.float32).to(device), test_inflow_Y.type(torch.float32).to(device)
		test_outflow_X, test_outflow_Y = test_outflow_X.type(torch.float32).to(device), test_outflow_Y.type(torch.float32).to(device)

		target = model(test_inflow_X, test_outflow_X, adjacency)

		loss = mse(input=test_inflow_Y, target=target)
		test_loss += loss.item()

		# evaluate on original scale
		# 获取result (batch, 276, pre_len)
		clone_prediction = target.cpu().detach().numpy().copy() * max_inflow  # clone(): Copy the tensor and allocate the new memory
		# print(clone_prediction.shape)  # (16, 276, 1)
		for i in range(clone_prediction.shape[0]):
			result.append(clone_prediction[i])

		# 获取result_original
		test_inflow_Y_original = test_inflow_Y_original.cpu().detach().numpy()
		# print(test_OD_Y_original.shape)  # (16, 276, 1)
		for i in range(test_inflow_Y_original.shape[0]):
			result_original.append(test_inflow_Y_original[i])

	print(np.array(result).shape, np.array(result_original).shape)  # (num, 276, 1)
	# 取整&非负取0
	result = np.array(result).astype(np.int)
	result[result < 0] = 0
	result_original = np.array(result_original).astype(np.int)
	result_original[result_original < 0] = 0
	# # 保存为一个npy文件
	# np.save("result/prediction/result.npy", np.array(result))
	# np.save("result/original/result_original.npy", np.array(result_original))
	# # 每一个时间步保存为一个OD矩阵
	# for i in range(np.array(result).shape[0]):
	# 	np.savetxt("result/prediction/" + str(i) + ".csv", result[i], delimiter=",")
	# 	np.savetxt("result/original/" + str(i) + "_original.csv", result_original[i], delimiter=",")
	#
	# # 取出多个车站进行画图   # (num, 276, 1)   # (num, 276, 2)  # (num, 276, 3)
	x = [[], [], [], [], []]
	y = [[], [], [], [], []]
	for i in range(result.shape[0]):
		x[0].append(result[i][4][0])
		y[0].append(result_original[i][4][0])
		x[1].append(result[i][18][0])
		y[1].append(result_original[i][18][0])
		x[2].append(result[i][30][0])
		y[2].append(result_original[i][30][0])
		x[3].append(result[i][60][0])
		y[3].append(result_original[i][60][0])
		x[4].append(result[i][94][0])
		y[4].append(result_original[i][94][0])
	result = np.array(result).reshape(station_num, -1)
	result_original = result_original.reshape(station_num, -1)

	RMSE, R2, MAE, WMAPE = Metrics(result_original, result).evaluate_performance()

	avg_test_loss = test_loss / len(inflow_data_loader_test)
	print('test Loss:', avg_test_loss)

	RMSE_y0, R2_y0, MAE_y0, WMAPE_y0 = Metrics_1d(y[0], x[0]).evaluate_performance()
	RMSE_y1, R2_y1, MAE_y1, WMAPE_y1 = Metrics_1d(y[1], x[1]).evaluate_performance()
	RMSE_y2, R2_y2, MAE_y2, WMAPE_y2 = Metrics_1d(y[2], x[2]).evaluate_performance()
	RMSE_y3, R2_y3, MAE_y3, WMAPE_y3 = Metrics_1d(y[3], x[3]).evaluate_performance()
	RMSE_y4, R2_y4, MAE_y4, WMAPE_y4 = Metrics_1d(y[4], x[4]).evaluate_performance()

# L3, = plt.plot(x[0], color="r")
# L4, = plt.plot(y[0], color="b")
# plt.legend([L3, L4], ["L3-prediction", "L4-true"], loc='best')
# plt.show()

ALL = [RMSE, MAE, WMAPE]
y0_ALL = [RMSE_y0, MAE_y0, WMAPE_y0]
y1_ALL = [RMSE_y1, MAE_y1, WMAPE_y1]
y2_ALL = [RMSE_y2, MAE_y2, WMAPE_y2]
y3_ALL = [RMSE_y3, MAE_y3, WMAPE_y3]
y4_ALL = [RMSE_y4, MAE_y4, WMAPE_y4]

np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_ALL.txt', ALL)
np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y0_ALL.txt', y0_ALL)
np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y1_ALL.txt', y1_ALL)
np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y2_ALL.txt', y2_ALL)
np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y3_ALL.txt', y3_ALL)
np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y4_ALL.txt', y4_ALL)
np.savetxt('result/X_original.txt', x)
np.savetxt('result/Y_prediction.txt', y)

print("ALL:", ALL)
print("y0_ALL:", y0_ALL)
print("y1_ALL:", y1_ALL)
print("y2_ALL:", y2_ALL)
print("y3_ALL:", y3_ALL)
print("y4_ALL:", y4_ALL)

print("end")

x = x[0]
y = y[0]
L1, = plt.plot(x, color="r")
L2, = plt.plot(y, color="y")
plt.legend([L1, L2], ["pre", "actual"], loc='best')
plt.show()
