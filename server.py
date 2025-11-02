import os
import random
import numpy as np
import pickle,re
import math
try:
    from bitarray import bitarray
except ImportError:
    bitarray = None


class Server(object):
	
	def __init__(self, conf, eval_dataset,hash_size=8,num_hashtable=3):
	
		self.conf = conf 

		self.w = self.conf["w"]

		self.hash_size = hash_size

		self.eval_dataset = eval_dataset

		self.hash_table_all=[]

		self.num_hashtable 	= num_hashtable

		self.b= [[random.uniform(0, self.w) for _ in range(self.hash_size)] for _ in range(self.num_hashtable)]#shape of b is (num_tables,t)

	def model_aggregate(self, hash_table=None,client_id=None,first_run=False):
		"""aggregate the hash table
		shape of hash_table is (num_tables,t,m)
		"""
		if first_run:
			hash_table_all = []
			for i in range(self.num_hashtable):
				b = np.array(self.b[i])#shape of b is (t,)
				keys = np.floor(
					(hash_table[i]+ b) / self.w)#shape of keys is (m,t)
				hash_table_all.append(keys)   #shape of hash_table_all is (num_tables,m,t)
				print(f"{client_id} hash_table {i} finished")
			self.hash_table_all.append(hash_table_all)
			with open(f"server_data/hash_table_all-{self.hash_size}-{self.num_hashtable}-{client_id}", "wb") as f:
				pickle.dump(hash_table_all, f)
		else:
			num=0
			for file_name in os.listdir("server_data"):
				if file_name.startswith("hash_table_all"):
					with open(f"server_data/hash_table_alll-{self.hash_size}-{self.num_hashtable}-{num}", "rb") as f:
						hash_table_all = pickle.load(f)
						self.hash_table_all.append(hash_table_all)
						num+=1

	def _hash(self, input_point_hash,b):
		""" Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A Python tuple or list object that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """

		try:
			projections = np.floor((input_point_hash + b) / self.w)
		except TypeError as e:
			print("""The input point needs to be an array-like object with
	              numbers only elements""")
			raise
		except ValueError as e:
			print("""The input point needs to be of the same dimension as
	              `input_dim` when initializing this LSHash instance""", e)
			raise
		else:
			return projections

	def query(self,query_point_hash_o, num_results=4, distance_func=None, threhold=None):
		""" Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point_hash:
            A list, or tuple, or numpy ndarray that only contains numbers.
            the dimension is (num_hash_tables,t)
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """
		if threhold == None:
			self.threhold = self.w ** 2
		else:
			self.threhold = threhold
		if not distance_func:
			distance_func = "euclidean"

		if distance_func == "hamming":
			if not bitarray:
				raise ImportError(" Bitarray is required for hamming distance")

		else:

			if distance_func == "euclidean":
				d_func = Server.euclidean_dist_square
			elif distance_func == "true_euclidean":
				d_func = Server.euclidean_dist
			elif distance_func == "centred_euclidean":
				d_func = Server.euclidean_dist_centred
			elif distance_func == "cosine":
				d_func = Server.cosine_dist
			elif distance_func == "l1norm":
				d_func = Server.l1norm_dist
			else:
				raise ValueError("The distance function name is invalid.")
			# print("querying is begin...")
			query_point_hash = []
			for i in range(self.num_hashtable):
				query_point_hash.append(self._hash(query_point_hash_o[i], self.b[i]))
			candicates={}
			for _ in range(100):
				distance_all, distance_sum, distance_count = self.calculate_candicates(query_point_hash,d_func=d_func,threhold=self.threhold)
				if len(distance_sum)>num_results:
					break
				else:
					slice_length = num_results * 3 if len(distance_all) >= num_results * 3 else len(distance_all)
					sliced_distance = list(distance_all)[:slice_length]
					# 这里以计算中位数为例，你可以根据需要修改为其他统计量，如均值 np.mean(sliced_distance) 或某个分位数 np.quantile(sliced_distance, 0.9)
					self.threhold = np.median(sliced_distance)
					# print(f"New threhold: {self.threhold}")
					continue

			candicates.update( {key: distance_sum[key] / distance_count[key] for key in distance_sum})
			candicates = [(ix,value) for (ix,value) in candicates.items()]
			# ix is the (client_id,data_line_number) value is the distance
			candicates.sort(key=lambda x: x[1])

		return candicates[:num_results] if num_results else candicates

	def calculate_candicates(self,query_point_hash, d_func=None, threhold=None):
			distance_sum = {}
			distance_count = {}
			distance_all=set()
			for filename in os.listdir("server_data"):
				if re.match(f"hash_table_all-{self.hash_size}-{self.num_hashtable}-\d+", filename):
					# print(f"read file {filename}")
					with open(os.path.join("server_data", filename), "rb") as f:
						hash_tables = pickle.load(f)#shape of hash_tables is (num_tables,m,t)
						client_id = int(filename.split(f"hash_table_all-{self.hash_size}-{self.num_hashtable}-")[1])
						# the shape of hash_tables is (num_tables,t,m)
						for i, table in enumerate(hash_tables):
							# i is the table_number
							# the shape of table is (m,t)
							# query_point_hash is (num_hash_tables,t)
							binary_hash = query_point_hash[i] # shape of binary_hash is (t,)
							for i_i,j in enumerate(table):#the i_i is the number of data in real data
								# j is the hash value of the data,the shape of j is (t,)
								distance = d_func(binary_hash, j)
								distance_all.add(distance)
								if distance < threhold:
									# i is the number of hash_tables
									# i_i is the number of data for getting the number of data in real data
									# print(f"client{client_id} hash_table{i} number{i_i} verified, distance is {distance}")
									key =(client_id,i_i)
									# print(key)
									if key not in distance_sum:
										distance_sum[key] = 0
										distance_count[key] = 0
									distance_sum[key] += distance
									distance_count[key] += 1
			return distance_all,distance_sum,distance_count

	### distance functions

	@staticmethod
	def hamming_dist(bitarray1, bitarray2):
		xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
		return xor_result.count()

	@staticmethod
	def euclidean_dist(x, y):
		""" This is a hot function, hence some optimizations are made. """
		diff = np.array(x) - y
		return np.sqrt(np.dot(diff, diff))

	@staticmethod
	def euclidean_dist_square(x, y):
		""" This is a hot function, hence some optimizations are made. """
		diff = np.array(x) - y
		return np.dot(diff, diff)

	@staticmethod
	def euclidean_dist_centred(x, y):
		""" This is a hot function, hence some optimizations are made. """
		diff = np.mean(x) - np.mean(y)
		return np.dot(diff, diff)

	@staticmethod
	def l1norm_dist(x, y):
		return sum(abs(x - y))

	@staticmethod
	def cosine_dist(x, y):
		return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)