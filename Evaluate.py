import torch
from datasetloader import DatasetLoader

MODEL_STORE_PATH = "Model"
model            = torch.load(MODEL_STORE_PATH)
model.eval()




# Min and max value of each feature
min_ip_address = 0
max_ip_address = 255255255255

min_port       = 0
max_port       = 65535

max_pkt_length = 1500 * 3 # * 3
min_pkt_length = 0

min_protocol   = 0
max_protocol   = 17

# Bandwidth = 10Gbits/10, pkt_size = 64 bits
min_num_pkt_per_s = 0
max_num_pkt_per_s = 125e+7

min_syn_flag      = 0
max_syn_flag      = 1

min_ack_flag      = 0
max_ack_flag      = 1

min_max_vect  = [
					[min_ip_address, max_ip_address], 
					[min_port, max_port], 
					[min_ip_address, max_ip_address], 
					[min_port, max_port], 
					[min_protocol, max_protocol], 
					[min_num_pkt_per_s, max_num_pkt_per_s],
					[min_num_pkt_per_s, max_num_pkt_per_s],
					[0, 1],
					[min_syn_flag, max_syn_flag],
					[min_ack_flag, max_ack_flag]
				]


features_list = [2, 3, 4, 5, 6, 43, 44, 50, 51, 54]
csv_syn       = '/home/kabre/Documents/Master-2/Projets/CNN/Dataset/Syn.csv'
csv_udp       = '/home/kabre/Documents/Master-2/Projets/CNN/Dataset/UDPLag.csv'

syn_attack    = 1
udp_attack    = 2

#- Load the dataset
data_syn      = DatasetLoader(csv_syn, syn_attack, 50000, 11000, features_list)
data_udp      = DatasetLoader(csv_udp, udp_attack, 100000, 1000, features_list)

syn_train, syn_labels     = data_syn.load_data()
udp_train, udp_labels     = data_udp.load_data()
train_data, train_labels  = data_syn.concat(udp_train, udp_labels)

print("\033[1m")
processing = 0
print("Normalization Process Starting")
for element in train_data:
	if processing == int (train_data.size(0) / 2):
		print("\033[32m Normalization {:.1f}%".format((processing / train_data.size(0))* 100))
	elif processing == train_data.size(0) - 2:
		print("\033[32m Normalization {:.0f}%".format((processing / train_data.size(0))* 100))
	processing = processing + 1
	for elmt in element:
		for vect in elmt:
			# Normalisation of the vector
			for ite in range(vect.size(0)):
				norme_vect= min_max_vect[ite]
				vect[ite] = DatasetLoader.normalize(vect[ite], norme_vect[0], norme_vect[1])
print(" End of Normalization")
print("\033[0m")
print("\033[1m")
print('Evaluation Starting')
data_length= train_data.size(0)
with torch.no_grad():
	correct = 0
	total   = 0
	pos = 0
	while pos < data_length:
		label        = train_labels[pos:pos+10]
		label        = label.long()
		output       = model(train_data[pos:pos+10])
		_, predicted = torch.max(output.data, 1)
		total       += label.size(0)
		correct     += (predicted == label).sum().item()
		pos += 10

	print(" Accuracy Over {} Packets : {:.1f}%".format(train_data.size(0) * 10, (correct / total) * 100))