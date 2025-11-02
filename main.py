import argparse, json
import datetime
import os
import logging

import numpy as np
import torch, random

from server import *
from client import *
import models, datasets
from sklearn.model_selection import train_test_split

	

if __name__ == '__main__':

	# parser = argparse.ArgumentParser(description='Federated Learning')
	# parser.add_argument('-c', '--conf', dest='conf')
	# args = parser.parse_args()

	# with open(args.conf, 'r') as f:
	# 	conf = json.load(f)
	with open("utils/conf.json", 'r') as f:
		conf = json.load(f)

	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	num_hash_tables = conf["hashtable_num"]
	hash_size = conf["hash_size"]
	input_dim = train_datasets.shape[1]
	train_dataset=[]
	chunks = np.array_split(train_datasets[0], conf["no_models"], axis=0)
	chunks_label = np.array_split(train_datasets[1], conf["no_models"], axis=0)
	w=conf["w"]
	# train_x2, valid_x2, train_y2, valid_y2 = train_test_split(imag_train, label_train, train_size=dim / 50000,
	# 														  random_state=8, stratify=label_train)

	def gen_uniform_planes(num_hash_tables,hash_size,input_dim):
		"""generate the w
		w: the dimension is (t,d) and the number is hash_table is num_hash_tables,
		uniform_planes:  (num_hash_tables,t,d),t is the hash_size, d is the input_dim such as cifar10 3072"""
		uniform_planes = [np.random.randn(hash_size, input_dim) for _ in range(num_hash_tables)]
		return uniform_planes

	uniform_planes = gen_uniform_planes(num_hash_tables,hash_size,input_dim)
	
	server = Server(conf, eval_datasets)
	clients = []

	chunks_id=0
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, (chunks[chunks_id],chunks_label[chunks_id]),uniform_planes, c))
		chunks_id+=1
	print("\n\n")
	chunks_id=0
	for c in clients:
		x_train_cyber,w_cyber = c.local_train()  #x_train_cyber is (m,d), w_cyber is (num_hash_tables,t,d)
		hash_table_cy = []
		# 将数据移动到 GPU 上
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		x_train_cyber_tensor = torch.tensor(x_train_cyber, dtype=torch.float32).to(device)
		for i in w_cyber:  # shape of i is (t,d), x is (m,d),the result is (t,m)
			i_tensor = torch.tensor(i, dtype=torch.float32).to(device)
			# 使用 PyTorch 进行矩阵乘法
			hash_table_temp = torch.matmul(i_tensor,
										   x_train_cyber_tensor.t())  # (t,d) * (d,m)  shape of hash_table_temp is (t,m)
			hash_table_cy.append(hash_table_temp.cpu().numpy())
		# for i in w_cyber:#shape of i is (t,d), x is (m,d),the result is (t,m)
		# 	hash_table_temp=np.dot(i,np.transpose(x_train_cyber))#(t*d)*(d,m)  shape of hash_table_temp is (t,m)
		# 	hash_table_cy.append(hash_table_temp)
		hash_table=c.gen_hash_table(hash_table_cy)  #shape of hash_table is (num_tables,t,m)
		server.model_aggregate(hash_table,clients.client_id)
		print("Node %d finished training" % c.client_id)

	def query(query_point):
		"""query the hash table"""
		#uniform_planes is (num_hash_tables,t,d)
		try:
			projections = []
			for i in uniform_planes:#the shape of i is (t,d)  the shape of query_point is (d,)
				projections.append(np.dot(i, query_point)) #projections is (num_hash_tables,t)
		except TypeError as e:
			print("""The input point needs to be an array-like object with
						 numbers only elements""")
			raise
		except ValueError as e:
			print("""The input point needs to be of the same dimension as
						 `input_dim` when initializing this LSHash instance""", e)
			raise
		else:
			server.query(projections)
			return tuple(projections)

	def accuracy(lsh,y,test_data,test_y,max_num=5):
		total=0
		correct=0
		for i in range(len(test_data)):
			result=lsh.query(test_data[i],max_num)
			if i%100==0:
				print("epoch {}:{}".format(i,len(result)))
			if len(result)>0:
				for hash_result in result:
					if y[hash_result[0][1]]==test_y[i]:
						correct=correct+1
					total=total+1
		return correct/total
	# for e in range(conf["global_epochs"]):
	#
	# 	candidates = random.sample(clients, conf["k"])
	#
	# 	weight_accumulator = {}
	#
	# 	for name, params in server.global_model.state_dict().items():
	# 		weight_accumulator[name] = torch.zeros_like(params)
	#
	# 	for c in candidates:
	# 		diff = c.local_train(server.global_model)
	#
	# 		for name, params in server.global_model.state_dict().items():
	# 			weight_accumulator[name].add_(diff[name])
	#
	#
	# 	server.model_aggregate(weight_accumulator)
	#
	# 	acc, loss = server.model_eval()
	#
	# 	print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
				
			
		
		
	
		
		
	