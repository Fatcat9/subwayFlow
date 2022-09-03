# _*_coding:utf-8_*_
import os
import time
import numpy as np

# 以两天的数据为例，共计276个车站，15分钟时间粒度下，每天05:00-23:00共计18个小时，每天共72个时间片
# 两天共计144个时间片，所以最终得到的为276×144的矩阵

global_start_time = time.time()
print(global_start_time)

# 导入n天的客流数据
datalist = os.listdir('./data/')
print(datalist)
datalist.sort(key=lambda x: int(x[9:13]))
# 初始化n个空列表存放n天的数据
for i in range(0, len(datalist)):
	globals()['flow_'+str(i)] = []

for i in range(0, len(datalist)):
	file = np.loadtxt('./data/'+ datalist[i], skiprows=1, dtype=str)
	for line in file:
		line = line.replace('"', '').strip().split(',')
		line = [int(x) for x in line]
		globals()['flow_'+str(i)].append(line)
	print("已导入第"+str(i)+"天的数据"+"  "+datalist[i])


# 获取车站在所给时间粒度下的进站客流序列
def get_tap_in(flow, time_granularity, station_num):
	# 一天共计1440分钟，去掉23点到5点五个小时300分钟的时间，一天还剩1080分钟,num为每天的时间片个数，
	# 当除不尽时，由于int是向下取整，会出现下标越界，所以加1
	if 1080 % time_granularity == 0:
		num = int(1080/time_granularity)
	else:
		num = int(1080/time_granularity)+1
	# 初始化278*278*num的多维矩阵,每个num代表第num个时间粒度
	OD_matrix = [[([0] * station_num) for i in range(station_num)] for j in range(num)]
	#print (matrix)
	for row in flow:
		# 每一列的含义 GRANT_CARD_CODE	TAP_IN	TAP_OUT	TIME_IN	TIME_OUT
		# row[1]为进站编码，row[2]为出站编码，row[3]为进站时间，t为进站时间所在的第几个时间粒度（角标是从0开始的所以要减1）
		# 通过row[3]将晚上11点到12点的数据删掉不予考虑
		if row[3] < 1380 and row[1] < 277 and row[2] < 277:
			m = int(row[1])-1
			n = int(row[2])-1
			t = int((int(row[3])-300)/time_granularity)+1
			# 对每一条记录，在相应位置进站量加1
			OD_matrix[t-1][m][n] += 1

	# 不同时间粒度下某个站点的进站量num列，行数为station_num
	O_matrix = [([0] * num) for i in range(station_num)]
	for i in range(num):
		for j in range(station_num):
			temp = sum(OD_matrix[i][j])
			O_matrix[j][i] = temp
	return O_matrix, OD_matrix


for i in [5, 10, 15, 30, 60]:
	print('正在提取第'+str(i)+'个时间粒度的时间序列')
	for j in range(len(datalist)):
		print('正在提取该时间粒度下第'+str(j)+'天的时间序列')
		globals()['O_flow_'+str(i)],  globals()['OD_matrix_'+str(i)] = get_tap_in(globals()['flow_'+str(j)], i, station_num=276)
		np.savetxt('O_flow_'+str(i)+'.csv', np.array(globals()['O_flow_'+str(i)]), delimiter=',', fmt='%i')
		print(globals()['O_flow_'+str(i)])

print('总时间为(s):', time.time() - global_start_time)
