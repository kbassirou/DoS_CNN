import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random as rd
import warnings
warnings.filterwarnings('ignore')


class DatasetLoader(Dataset):

	def __init__(self, csv_file, type_attack, num_rows, skrows, *columns):

		super(DatasetLoader, self).__init__()
		"""
		Args:
			csv_file (string) : Path to the csv_file with all packet features
			type_attack (int) : The type of the attack in range of [1..5]
			columns (list)    : List of columns to select in the csv file
		"""
		self._csv = csv_file
		self._window = 10
		self._type_attack = type_attack
		self._skr = skrows
		self._numrows = num_rows
		self._train = []
		self._labels= []

		if type(columns) is tuple and len(columns) > 0:
			first_element  = columns[0]
			self._features = list(first_element)
			self._dataset  = pd.read_csv(csv_file, nrows=num_rows, skiprows=skrows, usecols=self._features)
		else:
			self._features = list()
			self._dataset  = pd.read_csv(csv_file,  nrows=num_rows, skiprows=skrows)


	# --

	def load_data(self):

		"""
			@return (torch.Tensor) : Transform a collection of matrix MxN in a pytorch tensor 
			in format (batch, in_channels, M, N) with batch = number of matrix in the collection
		"""
		all_data       = self._dataset.iloc[:, :].as_matrix()
		number_of_line = len(all_data)
		dataset        = list()
		position       = 0
		batch          = 0
		while position <= number_of_line - batch:
			batch       += 1
			current_pkt = all_data [position:position+self._window, :]

			for each_pkt in current_pkt:
				for ite in range(len(each_pkt)):
					if type(each_pkt[ite]) is str:
						s_list = each_pkt[ite].split('.')
						s_list = ''.join(s_list)
						s_list = int(s_list)
						each_pkt[ite] = s_list

			dataset.append(current_pkt)
			position    += self._window
			
		# Creation of the associated labels
		labels  = [self._type_attack for ite in range(batch)]
		labels  = torch.Tensor(labels)
		dataset = torch.Tensor(dataset)
		dataset = dataset.reshape(batch, 1, 10, len(self._features))
		self._train = dataset
		self._labels= labels
		return dataset, labels


	# --

	def concat(self, sample, labels, shuffle=False):
		"""
		Args:
			s_sample (torch.Tensor) : PyTorch in this format (batch, in_channels, M, N)
			batch (number of matrix), M (number of line in each matrix), N (number of 
			columns in each matrix)

		Return:
			dst (torch.Tensor) : Concate of self._train and sample
		"""
		s_dst = torch.cat((self._train, sample), dim=0)
		l_dst = torch.cat((self._labels, labels), dim=0)       
		return s_dst, l_dst


	# --

	
	def normalize(cls, x, xmin, xmax):
		"""
		Args:
			x (int || float)    : Value to normalize
			xmin (int || float) : The possible minimum value of x
			xmax (int || float) : The possible maximum value of x
		"""

		if x == 0.0:
			return 0
		else:
			return  (x - xmin) / (xmax - xmin)


	normalize = classmethod(normalize)
