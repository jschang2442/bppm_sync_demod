import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

# Unpack the data
def prepare_data(filename):
	name = "data/" + filename
	df = pd.read_csv(name)
	df.columns = ['measurements', 'encodings']
	my_measurements = torch.tensor(df['measurements'].values)
	my_encodings = torch.tensor(df['encodings'].values)
	return my_measurements, my_encodings

def batchify_data(x_data, y_data, batch_size):
	"""Takes a set of data points and labels and groups them into batches."""
	# Only take batch_size chunks (i.e. drop the remainder)
	N = int(len(x_data) / batch_size) * batch_size
	batches = []
	for i in range(0, N, batch_size):
		batches.append({
			'x': torch.tensor(x_data[i:i+batch_size], dtype=torch.float32),
			'y': torch.tensor(y_data[i:i+batch_size], dtype=torch.long
		)})
	return batches

def compute_accuracy(predictions, y):
	return np.mean(np.equal(predictions.numpy(), y.numpy()))

def train_model(train_data, val_data, model, learning_rate, num_epochs = 2):
# add num_epochs?
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		print("-------------\nEpoch {}:\n".format(epoch))


		# Run **training***
		loss, acc = run_epoch(train_data, model.train(), optimizer)
		print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))

		# Run **validation**
		val_loss, val_acc = run_epoch(val_data, model.eval(), optimizer)
		print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
		# Save model
		torch.save(model, 'sym_synch_lstm.pt')
	return val_acc

def run_epoch(data, model, optimizer):
	losses = []
	batch_accuracies = []

	is_training = model.training

	for batch in tqdm(data):
		x, y = batch['x'], batch['y']

		#x = x.view([1, 1, 200])
		#y = y.view([1, 1, 200])

		model.hidden = model.init_hidden()
		#pdb.set_trace()
		out = model(x)

		predictions = torch.argmax(out)
		batch_accuracies.append(compute_accuracy(predictions, y))

		loss = F.cross_entropy(out.view(1, 2), y)
		losses.append(loss.data.item())

		if is_training:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	avg_loss = np.mean(losses)
	avg_accuracy = np.mean(batch_accuracies)

	return avg_loss, avg_accuracy



# Define the model
class myLSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim = 1, num_layers=2):
		super(myLSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
		self.linear = nn.Linear(self.hidden_dim, output_dim)

	def init_hidden(self):
		return (torch.zeros(self.num_layers, 1, self.hidden_dim),
				torch.zeros(self.num_layers, 1, self.hidden_dim))

	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input.view(len(input), 1, -1))
		y_pred = self.linear(lstm_out[-1].view(1, -1))
		return y_pred.view(-1)



if __name__ == '__main__':
	x, y = prepare_data("ml_project_sps10_snr49_seed0.csv")
	BATCH_SIZE = 1 # this is the number of samples per batch
	train_batches = batchify_data(x, y, BATCH_SIZE)

	x, y = prepare_data("ml_project_sps10_snr49_seed1.csv")
	test_batches = batchify_data(x, y, BATCH_SIZE)

	#batches is a list of x, y pairs. 

	#pdb.set_trace()
	
	plt.plot(y.detach().numpy())
	plt.plot(x.detach().numpy())
	plt.show()

	np.random.seed(0)
	torch.manual_seed(0)
	NUM_TRAIN = 2
	NUM_TEST = 2
	NUM_CLASSES = 2
	lstm_input_size = 1

	#size of hidden layers
	h1 = 32
	output_dim = 2
	NUM_LAYERS = 30
	learning_rate = 1e-3
	num_epochs = 20

	model = myLSTM(lstm_input_size, h1, output_dim = output_dim, num_layers = NUM_LAYERS)
	# 
	# x_train = 
	# x_test = 
	# y_train = 
	# y_test = 

	train_model(train_batches, test_batches, model, learning_rate = learning_rate, num_epochs = num_epochs)



	
# need offset for batchify
# 