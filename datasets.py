import torch
import numpy as np
import pickle
from torchvision import datasets, transforms
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def get_dataset(dir, name,sequence_length=21,overlap=100):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
	elif name=="cifar10":
		dict1 = unpickle('./data/cifar-10-batches-py/data_batch_1')
		dict2 = unpickle('./data/cifar-10-batches-py/data_batch_2')
		dict3 = unpickle('./data/cifar-10-batches-py/data_batch_3')
		dict4 = unpickle('./data/cifar-10-batches-py/data_batch_4')
		dict5 = unpickle('./data/cifar-10-batches-py/data_batch_5')
		test_data = unpickle('./data/cifar-10-batches-py/test_batch')
		imag_train = np.vstack((dict1[b'data'], dict2[b'data'], dict3[b'data'], dict4[b'data'], dict5[b'data']))
		label_train = dict1[b'labels'] + dict2[b'labels'] + dict3[b'labels'] + dict4[b'labels'] + dict5[b'labels']
		imag_test = test_data[b'data']
		label_test = test_data[b'labels']
		train_dataset=(imag_train,label_train)
		eval_dataset=(imag_test,label_test)
	elif name=="ncbi":
		from Bio import SeqIO
		# 读取FNA文件
		file_path = "data/ncbi_dataset/GCA_027704885.1_ASM2770488v1_genomic.fna"
		sequences = list(SeqIO.parse(file_path, "fasta"))

		def integer_encode(sequence):
			base_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
			encoded = [base_to_int.get(base, -1) for base in sequence]  # -1处理未知碱基（如N）
			return encoded
		data = []
		label= []
		j = 0
		count=0
		file_name=0
		for i in sequences:
			encoded_seq = integer_encode(str(i.seq))
			n_chunks= (len(encoded_seq ) - overlap) // (sequence_length - overlap)
			for chunk_i in range(n_chunks):
				start = chunk_i * (sequence_length - overlap)
				end = start + sequence_length
				kmer = encoded_seq[start:end]
				data.append(kmer)
				label.append(j)
				count=count+1
				if (count+1)%10000==0:
					with open(f"data/ncbi_batch/ncbi_batch_{sequence_length}_{overlap}_{file_name}","wb") as f:
						train_dataset=(np.array(data),label)
						pickle.dump(train_dataset,f)
						data=[]
						label=[]
						print(f"ncbi_batch_{file_name}")
						file_name=file_name+1
				print(f"ncbi {j} sequence chunk {chunk_i}")
			# 处理剩余部分（可选填充）
			remaining = len(encoded_seq) - n_chunks * (sequence_length - overlap)
			if remaining > 0:
				last_chunk = encoded_seq[-sequence_length:] if len(encoded_seq) >= sequence_length else encoded_seq + [-1] * (sequence_length - len(encoded_seq))
				data.append(last_chunk)
				label.append(j)
				count=count+1
				chunk_i=chunk_i+1
				if (count+1)%10000==0:
					with open(f"data/ncbi_batch/ncbi_batch_{sequence_length}_{overlap}_{file_name}","wb") as f:
						train_dataset=(np.array(data),label)
						pickle.dump(train_dataset,f)
						data=[]
						label=[]
						print(f"ncbi_batch_{file_name}")
						file_name=file_name+1
				print(f"ncbi {j} sequence chunk {chunk_i}")
			j = j + 1
		if len(data)!=0:
			with open(f"data/ncbi_batch/ncbi_batch_{sequence_length}_{overlap}_{file_name+1}","wb") as f:
				train_dataset=(np.array(data),label)
				pickle.dump(train_dataset,f)
				print(f"ncbi_batch_{file_name}")
		print(f"ncbi {j} sequence chunk {chunk_i+1}")
		print("total_count:",count)
		train_dataset=eval_dataset=[]
	return train_dataset, eval_dataset