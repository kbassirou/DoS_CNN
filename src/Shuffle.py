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
	
	if shuffle is True:
		tmp1 = 0
		tmp2 = 0
		s_dst_size = s_dst.size(0)
		idx1_list  = []
		idx2_list  = []
		for ite in range(int (s_dst_size / 2)):
			idx1 = int ( rd.random() * s_dst_size)
			while idx1 in idx1_list:
				idx1 = int ( rd.random() * s_dst_size )
			idx2 = int ( rd.random() * s_dst_size)
			while idx2 in idx2_list:
				idx2 = int ( rd.random() * s_dst_size )

			idx2_list.append(idx2)
			idx1_list.append(idx1)

			# Shuffle the labels list
			tmp1 = l_dst[idx1]
			tmp1 = tmp1.item()
			l_dst[idx1] = l_dst[idx2]
			l_dst[idx2] = tmp1

			# Shuffle the sample list
			tmp2 = s_dst[idx1]
			s_dst[idx1] = s_dst[idx2]
			s_dst[idx2] = tmp2

                                        
	return s_dst, l_dst