from data.datasets import Traffic_inflow
from torch.utils.data import DataLoader


inflow_data = "./data/in_15min.csv"
outflow_data = "./data/out_15min.csv"


def get_inflow_dataloader(time_interval=30, time_lag=5, tg_in_one_day=36, forecast_day_number=5, pre_len=1, batch_size=8):
	# train inflow data loader
	print("train inflow")
	inflow_train = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number,
								pre_len=pre_len, inflow_data=inflow_data, is_train=True, is_val=False, val_rate=0.1)
	max_inflow, min_inflow = inflow_train.get_max_min_inflow()
	inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

	# validation inflow data loader
	print("val inflow")
	inflow_val = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number,
								pre_len=pre_len, inflow_data=inflow_data, is_train=True, is_val=True, val_rate=0.1)
	inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

	# test inflow data loader
	print("test inflow")
	inflow_test = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number,
								pre_len=pre_len, inflow_data=inflow_data, is_train=False, is_val=False, val_rate=0)
	inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

	return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow


def get_outflow_dataloader(time_interval=15, time_lag=5, tg_in_one_day=72, forecast_day_number=5, pre_len=1, batch_size=8):
	# train inflow data loader
	print("train outflow")
	inflow_train = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number,
								pre_len=pre_len, inflow_data=outflow_data, is_train=True, is_val=False, val_rate=0.1)
	max_inflow, min_inflow = inflow_train.get_max_min_inflow()
	inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

	# validation inflow data loader
	print("val outflow")
	inflow_val = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number,
								pre_len=pre_len, inflow_data=outflow_data, is_train=True, is_val=True, val_rate=0.1)
	inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

	# test inflow data loader
	print("test outflow")
	inflow_test = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, forecast_day_number=forecast_day_number,
								pre_len=pre_len, inflow_data=outflow_data, is_train=False, is_val=False, val_rate=0)
	inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

	return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow
