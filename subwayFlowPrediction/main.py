import numpy as np
import os, time, torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import GetLaplacian
from model.main_model import Model
from utils.earlystopping import EarlyStopping
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


# 用于初始化卷积层的参数，可提升模型训练效果
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.0)
	if classname.find('ConvTranspose2d') != -1:
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.0)


model = Model(time_lag, pre_len, station_num, device)
print(model)
model.apply(weights_init)
if torch.cuda.is_available():
	model.cuda()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

temp_time = time.time()
early_stopping = EarlyStopping(patience=100, verbose=True)
for epoch in range(0, epoch_num):
	# model train
	train_loss = 0
	model.train()
	for inflow_tr, outflow_tr in zip(enumerate(inflow_data_loader_train), enumerate(outflow_data_loader_train)):
		i_batch, (train_inflow_X, train_inflow_Y) = inflow_tr
		i_batch, (train_outflow_X, train_outflow_Y) = outflow_tr
		train_inflow_X, train_inflow_Y = train_inflow_X.type(torch.float32).to(device), train_inflow_Y.type(torch.float32).to(device)
		train_outflow_X, train_outflow_Y = train_outflow_X.type(torch.float32).to(device), train_outflow_Y.type(torch.float32).to(device)
		target = model(train_inflow_X, train_outflow_X, adjacency)
		loss = mse(input=train_inflow_Y, target=target)
		train_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	with torch.no_grad():
		# model validation
		model.eval()
		val_loss = 0
		for inflow_val, outflow_val in zip(enumerate(inflow_data_loader_val), enumerate(outflow_data_loader_val)):
			i_batch, (val_inflow_X, val_inflow_Y) = inflow_tr
			i_batch, (val_outflow_X, val_outflow_Y) = outflow_tr
			val_inflow_X, val_inflow_Y = val_inflow_X.type(torch.float32).to(device), val_inflow_Y.type(torch.float32).to(device)
			val_outflow_X, val_outflow_Y = val_outflow_X.type(torch.float32).to(device), val_outflow_Y.type(torch.float32).to(device)
			target = model(val_inflow_X, val_outflow_X, adjacency)
			loss = mse(input=val_inflow_Y, target=target)
			val_loss += loss.item()

	avg_train_loss = train_loss/len(inflow_data_loader_train)
	avg_val_loss = val_loss/len(inflow_data_loader_val)
	writer.add_scalar("loss_train", avg_train_loss, epoch)
	writer.add_scalar("loss_eval", avg_val_loss, epoch)
	print('epoch:', epoch, 'train Loss', avg_train_loss, 'val Loss:', avg_val_loss)

	if epoch > 0:
		# early stopping
		model_dict = model.state_dict()
		early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
		if early_stopping.early_stop:
			print("Early Stopping")
			break
	# 每10个epoch打印一次训练时间
	if epoch % 10 == 0:
		print("time for 10 epoches:", round(time.time() - temp_time, 2))
		temp_time = time.time()
global_end_time = time.time() - global_start_time
print("global end time:", global_end_time)

Train_time_ALL = []

Train_time_ALL.append(global_end_time)
np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_Train_time_ALL.txt', Train_time_ALL)
print("end")
