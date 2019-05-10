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


def batchify_data(x_data, y_data, seq_length, batch_size=1):
        """Takes a set of data points and labels and groups them into batches."""
        # Only take batch_size chunks (i.e. drop the remainder)
        N = int(len(x_data) / seq_length) * seq_length
        batches = []
        for i in range(0, N, seq_length):
                batches.append({
                        'x': x_data[i:i+seq_length].view(-1, 1).clone().float(),
                        'y': y_data[i:i+seq_length].view(-1, 1).clone().float()
                })
        return batches


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


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

        # Initialize the hidden state for the *start* of the epoch.
        hidden = model.init_hidden()
        
        for batch in tqdm(data):
                # Expect x = batch size x seq length x num features
                # Expect y = batch size x seq length
                x, y = batch['x'], batch['y']

                # Get dimensions... and transpose.
                seq_len, nfeat = x.size()
                x = x.view(seq_len, 1, nfeat)
                
                # Pass in current x_{t} + h_{t-1}
                out, hidden = model(x, hidden)
                hidden = repackage_hidden(hidden)
                
                # Compute predictions...
                predictions = out.gt(0).float()
                batch_accuracies.append(compute_accuracy(predictions, y))

                # Compute the loss for every output in sequence.
                loss = F.binary_cross_entropy_with_logits(out.view(-1), y.view(-1))
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
        def __init__(self, input_dim, hidden_dim, num_layers=2):
                super(myLSTM, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
                self.linear = nn.Linear(self.hidden_dim, 1)

        def init_hidden(self, bsz=1):
                weight = next(self.parameters())
                return (weight.new_zeros(self.num_layers, bsz, self.hidden_dim),
                        weight.new_zeros(self.num_layers, bsz, self.hidden_dim))

        def forward(self, input, hidden):
                """
                input: batch size x seq length x features
                (expect features = 1)

                return: batch size x seq length
                """
                out, hidden = self.lstm(input, hidden)
                y_pred = self.linear(out).squeeze(-1)
                return y_pred, hidden



if __name__ == '__main__':
        x, y = prepare_data("ml_project_sps10_snr49_seed0.csv")
        BATCH_SIZE = 1 # this is the number of samples per batch
        SEQ_LENGTH = 64
        train_batches = batchify_data(x, y, SEQ_LENGTH, BATCH_SIZE)

        x, y = prepare_data("ml_project_sps10_snr49_seed1.csv")
        test_batches = batchify_data(x, y, SEQ_LENGTH, BATCH_SIZE)

        #batches is a list of x, y pairs. 

        #pdb.set_trace()
        
        # plt.plot(y.detach().numpy())
        # plt.plot(x.detach().numpy())
        # plt.show()

        np.random.seed(0)
        torch.manual_seed(0)
        NUM_TRAIN = 2
        NUM_TEST = 2
        NUM_CLASSES = 2
        lstm_input_size = 1

        #size of hidden layers
        h1 = 32
        output_dim = 2
        NUM_LAYERS = 1
        learning_rate = 1e-3
        num_epochs = 20

        model = myLSTM(lstm_input_size, h1, num_layers = NUM_LAYERS)
        print(model)
        # 
        # x_train = 
        # x_test = 
        # y_train = 
        # y_test = 

        train_model(train_batches, test_batches, model, learning_rate = learning_rate, num_epochs = num_epochs)



        
# need offset for batchify
# 
