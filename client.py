import random
import math
import numpy as np
import time,pickle
import scipy.sparse as sp
class Client(object):

	def __init__(self, conf, train_dataset, uniform_planes,id = -1,hash_size=8,num_hashtables=3):
		
		self.conf = conf
		
		self.client_id = id

		self.l = self.conf["l"]   #the number of Givens
		
		self.train_dataset = train_dataset[0]  # m*d

		print("shape of train_dataset is ",self.train_dataset.shape)

		self.train_label = train_dataset[1]

		self.hash_size = hash_size  #t

		self.batch_size = self.conf["batch_size"]

		self.num_hashtables = num_hashtables  #num_hash_tables

		self.uniform_planes = uniform_planes   # (num_hash_tables,t,d)
		
		# all_range = list(range(len(self.train_dataset)))
		# data_len = int(len(self.train_dataset) / self.conf['no_models'])
		# train_indices = all_range[id * data_len: (id + 1) * data_len]
		#
		# self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
		# 							sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
									
	def Givens_generation(self,dim, i, j):
		"""dim is the dimension of the matrix dim*dim
		i is the row index
		j is the column index is
		"""
		random.seed()
		rad = random.randint(0, int(2 * math.pi))
		c = math.cos(rad)
		s = math.sin(rad)
		matrix = [[0 for temp1 in range(dim)] for temp2 in range(dim)]
		for i_i in range(dim):
			if i_i == i or i_i == j:
				matrix[i_i][i_i] = c
			else:
				matrix[i_i][i_i] = 1
		print("i,j=",i,j)
		matrix[i][j] = s
		matrix[j][i] = -s
		return sp.csr_matrix(matrix)

	def secret_key(self, x, y, l):
		"""
		x: the dimension of m, if is the batch train, the shape of x is batch_size
		y: the dimension of d
		l: the number of Givens

		"""
		secret_k = []
		q = []
		p = []
		random.seed()
		for i in range(l):
			p.append(self.Givens_generation(x, random.randint(0, int(x / 2)-1), random.randint(int(x / 2), x-1)))
		for i in range(l):
			q.append(self.Givens_generation(y, random.randint(0, int(y / 2)-1), random.randint(int(y / 2), y-1)))
		secret_k.append(p) #shape of p is (l,m,m), batch is (l,batch_size,batch_size)
		secret_k.append(q)#shape of q is (l,d,d)
		secret_k.append(random.random() * 1000)#shape:(p,q,a)
		self.secret_k=secret_k

	def create_x_cy(self,train_x):
		"""x is m*d
		train_x_cy is m*d
		"""
		print("cyber x...")
		train_x_cy = self.secret_k[2] * train_x
		print("after a...")
		for p in self.secret_k[0]:
			train_x_cy = p @ train_x_cy
		print("after p...")
		for q in self.secret_k[1]:
			train_x_cy = train_x_cy @ q
		print("after q...")
		return train_x_cy

	def create_w_cy(self,w):
		w_cy = []
		for i in w:
			i_cy = self.secret_k[2] * i
			for q in self.secret_k[1]:
				i_cy = i_cy @ q
			w_cy.append(i_cy)
		return w_cy

	# def de_cy_hash(self,hash_table_cy):
	# 	"""decypher the hash table"""
	# 	hash_table = []
	# 	for i in range(len(hash_table_cy)):
	# 		b = np.array([random.uniform(0, 4) for j in range(len(hash_table_cy[i]))])
	# 		items = np.array(list(hash_table_cy[i]))
	# 		keys = np.transpose(items)
	# 		for p in self.secret_k[0]:
	# 			temp = np.dot(keys, np.array(p))
	# 		hash_table_temp = np.floor(
	# 			(np.transpose(temp) / (self.secret_k[2] ** 2) + np.transpose(b * np.ones((temp.shape[0], len(b))))) / 4)
	# 		hash_table.append(hash_table_temp)
	# 	return hash_table
	def de_cy_hash(self,hash_table_cy):
		"""decypher the hash table
		the shape of hash_table_cy is (num_tables,m,t)

		"""

		# if self.train_mode=="total":
		hash_table = []
		for i in range(self.num_hashtables):#shape of i is (m,t)
			items = np.array(hash_table_cy[i]) #shape of items is (m,t)
			temp = items #shape of keys is (t,m)
			for p in self.secret_k[0]:#shape of p is (m,m)
				temp = p.T @ temp #shape of temp is (m,t)
			temp = temp /(self.secret_k[2]**2)
			hash_table.append(temp)#shape of hash_table is (num_tables,m,t)
# 		elif self.train_mode=="batch":
# 			hash_table = [[] for _ in range(self.num_hashtables)]
# 			for batch_idx in range(self.num_batches):
# 				start_idx = batch_idx * self.batch_size
# 				end_idx = (batch_idx + 1) * self.batch_size

# 				for i in range(self.num_hashtables):#shape of i is (m,t)
# 					items = np.array(hash_table_cy[i]) [start_idx:end_idx]#shape of items is (batch_size,t)
# 					temp = items #shape of keys is (batch_size,t)
# 					for p in self.secret_k[0]:#shape of p is (batch_size,batch_size)
# 						temp = p @ temp  #shape of temp is (m,t)
# 					temp = temp /(self.secret_k[2]**2)
# 					hash_table[i].append(temp)#shape of hash_table is (num_tables,m,t)
# 				print(f"batch_hash_table recovered for batch {batch_idx}....")
# 			for i in range(self.num_hashtables):
# 				hash_table[i] = np.vstack(hash_table[i])

		return hash_table

	def verify(self, hash_cy,x_train_cy):
		r = [random.random() for i in range(self.hash_size)]# size of r is (t,)
		if self.w_cy is None:
			with open(f"cyber_data/w_cyber-{self.hash_size}-{self.num_hashtables}-{self.client_id}", "rb") as f:
				self.w_cy = pickle.load(f)
		for i in range(len(hash_cy)):
			v = np.dot(np.array(r), np.array(self.w_cy[i]))# size of v is (t,) size of w_cy is (num_hash_tables,t,d)
			v = np.dot(v, np.transpose(x_train_cy))# size of x_train_cy is (m,d) v is (t,m)
			v2 = np.dot(np.array(r), np.array(hash_cy[i]))# size of hash_cy[i] is (t,m)
			print("v:",v)
			print("v2:",v2)
			if (v == v2).all:
				return 1
			else:
				return 0

	def gen_hash_table(self,hash_table_cy,i=-1):
		"""generate the hash table"""
		if self.train_mode=="total":
			ver_flag=self.verify([table.T for table in hash_table_cy],self.x_train_cy) #shape of hash_table_cy is (num_tables,m,t))
			if ver_flag==0:
				print("hash table is not correct")
				return 0
			print("hash table is verified")
			hash_table = self.de_cy_hash(hash_table_cy)
			self.hash_table = hash_table
			return hash_table
		else:
			ver_flag=self.verify([table.T for table in hash_table_cy],self.x_train_cy[i]) #shape of hash_table_cy is (num_tables,m,t))
			if ver_flag==0:
				print("hash table is not correct")
				return 0
			print(f"hash table batch{i} is verified")
			hash_table = self.de_cy_hash(hash_table_cy)
			self.hash_table = hash_table
			return hash_table

	def local_train(self,mode="batch"):
		if mode == "total":
			self.train_mode="total"
			time_start = time.time()
			self.secret_key(self.train_dataset.shape[0], self.train_dataset.shape[1], self.l)
			print("secret_k generated....")
			self.x_train_cy = self.create_x_cy(self.train_dataset)
			print("x_train_cy generated....")
			self.w_cy = self.create_w_cy(np.array(self.uniform_planes))#(num_hash_tables,t,d)
			print("w_cy generated...")
			time_end = time.time()
			with open(f"cyber_data/x_train_cyber-{self.hash_size}-{self.num_hashtables}-{self.client_id}", "wb") as f:
				pickle.dump(self.x_train_cy, f)
			with open(f"cyber_data/w_cyber-{self.hash_size}-{self.num_hashtables}-{self.client_id}", "wb") as f:
				pickle.dump(self.w_cy, f)
			time_temp = time_end - time_start
			print(f"Node{self.client_id} time cost: {time_temp}")
			#time_dim_cost_all.append(time_dim_cost_all)
			return self.x_train_cy, self.w_cy
		elif mode == "batch":
			self.train_mode="batch"
			time_start = time.time()
			num_batches = len(self.train_dataset) // self.batch_size
			self.num_batches = num_batches
			all_x_train_cy = []

			self.secret_key(self.batch_size, self.train_dataset.shape[1], self.l)
			print(f"Secret key generated for total batch....")

			w_cy = self.create_w_cy(np.array(self.uniform_planes))# the shape of w_cy is (num_hash_tables,t,d)

			self.w_cy = w_cy
			print(f"w_cy generated")
			with open(f"cyber_data_batch/w_cyber{self.client_id}", "wb") as f:
				pickle.dump(w_cy, f)

			for batch_idx in range(num_batches):
				start_idx = batch_idx * self.batch_size
				end_idx = (batch_idx + 1) * self.batch_size
				batch_train_dataset = self.train_dataset[start_idx:end_idx]

				x_train_cy = self.create_x_cy(batch_train_dataset)
				print(type(x_train_cy))
				print(f"x_train_cy generated for batch {batch_idx}....")

				all_x_train_cy.append(x_train_cy)

				with open(f"cyber_data_batch/x_train_cyber{self.client_id}_batch{batch_idx}", "wb") as f:#Incremental stream processing
					pickle.dump(x_train_cy, f)

				# 处理最后一个可能不足 batch_size 的批次
			# if len(self.train_dataset) % self.batch_size != 0:
			# 	start_idx = num_batches * self.batch_size
			# 	batch_train_dataset = self.train_dataset[start_idx:]
			#
			# 	print(f"Secret key already generated for all batches....")
			# 	x_train_cy = self.create_x_cy(batch_train_dataset)
			# 	print(f"x_train_cy generated for the last batch....")
			# 	w_cy = self.create_w_cy(np.array(self.uniform_planes))
			# 	print(f"w_cy generated for the last batch...")
			#
			# 	all_x_train_cy.append(x_train_cy)
			# 	all_w_cy.append(w_cy)
			#
			# 	last_batch_idx = num_batches
			# 	with open(f"x_train_cyber{self.client_id}_batch{last_batch_idx}", "wb") as f:
			# 		pickle.dump(x_train_cy, f)
			# 	with open(f"w_cyber{self.client_id}_batch{last_batch_idx}", "wb") as f:
			# 		pickle.dump(w_cy, f)
			# 纵向合并 all_x_train_cy 和 all_w_cy
			self.x_train_cy = all_x_train_cy
			time_end = time.time()
			time_temp = time_end - time_start
			print(f"Node{self.client_id} time cost for batch mode: {time_temp}")

			return all_x_train_cy, np.array(w_cy)
		else:
			raise ValueError("The mode is invalid.")

