from datasetloader import DatasetLoader
from cnn import ConvNetDoS
import torch
import matplotlib.pyplot as plt


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

# 5

# Features columns number

#  Ip source        : 2
#  Port source      : 3
#  Ip destination   : 4
#  Port destination : 5
#  Protocole        : 6
#  Nombre de paquet dans une seconde  FWD(43)  BWD(44)
#  Nombre de paquet SYN  (51) Syn Flag Count
#  Nombre de paquet ACK  (54) ACK Flag Count
#  Taille de paquets  FWD - SYN_csv(11) BWd - SYN_csv(12)

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


MODEL_STORE_PATH = "Model"
model         = ConvNetDoS()
print("\033[1m")
print("Features Extraction")
features_list = [2, 3, 4, 5, 6, 43, 44, 50, 51, 54]
csv_syn       = '/home/kabre/Documents/Master-2/Projets/CNN/Dataset/Syn.csv'
csv_udp       = '/home/kabre/Documents/Master-2/Projets/CNN/Dataset/UDPLag.csv'

syn_attack    = 1
udp_attack    = 2

#- Load the dataset

data_syn      = DatasetLoader(csv_syn, syn_attack, 40, 0, features_list)
data_udp      = DatasetLoader(csv_udp, udp_attack, 40, 40,  features_list)
syn_train, syn_labels     = data_syn.load_data()
udp_train, udp_labels     = data_udp.load_data()
train_data, train_labels  = data_syn.concat(udp_train, udp_labels)

for i in range(5000):
	data_syn      = DatasetLoader(csv_syn, syn_attack, 10, i * 10, features_list)
	syn_train, syn_labels = data_syn.load_data()
	train_data, train_labels = data_syn.concat(train_data, train_labels)
	data_udp      = DatasetLoader(csv_udp, udp_attack, 10, i * 10,  features_list)
	udp_train, udp_labels     = data_udp.load_data()
	train_data, train_labels  = data_udp.concat(train_data, train_labels)


processing = 0

print("\033[32m End Of Extraction")
print("\033[0m", end=" ")
print("\033[1m")

# - Normalization 
print("Normalization Process Starting")
for element in train_data:
	if processing == int (train_data.size(0) / 2):
		print("\033[32m Normalization {:.1f}%".format((processing / train_data.size(0))* 100))
	elif processing == train_data.size(0) - 2:
		print("\033[32m Normalization {:.1f}%".format((processing / train_data.size(0))* 100), end=" ")
		print("\033[0m", end=" ")
	processing = processing + 1
	for elmt in element:
		for vect in elmt:
			# Normalisation of the vector
			for ite in range(vect.size(0)):
				norme_vect= min_max_vect[ite]
				vect[ite] = DatasetLoader.normalize(vect[ite], norme_vect[0], norme_vect[1])
print('\033[1m')
print('\033[32m Normalization Done !', end=' ')
print('\033[0m')

# - End of the normalization

criterion = torch.nn.CrossEntropyLoss()
# Loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print('\033[1m')
print('Training Process Starting')
num_epochs = 6
loss_list  = []
data_length= train_data.size(0)

epoch  = 0
begin  = 0
loss   = 0
while epoch < num_epochs:
	pos = 0
	if begin != 0:
		print(' Epoch [{}/{}], Loss : {:.4f}'.format(epoch, num_epochs, loss.item()))
		loss_list.append(loss.item())
	while pos <= data_length - 10:
		batch   = train_data[pos:pos+10]
		output  = model(batch)
		label   = train_labels[pos:pos+10]
		label   = label.long()
		loss    = criterion(output, label)
		pos    += 10
		# Backprop and perform SGD optimisation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	epoch = epoch + 1
	begin = 1
	
print('Training Finished !')
print('')

print('Saving The Model')
torch.save(model, MODEL_STORE_PATH)
print("\033[32m Saving Ended !")
print('\033[0m')
print('')

print('Plotting Loss Curve')
#epoch = [ite + 1 for ite in range(num_epochs)]
plt.plot(loss_list, 'r', label="Loss")
plt.title("Loss curve")
plt.xlabel("Epoch Number")
plt.show()
