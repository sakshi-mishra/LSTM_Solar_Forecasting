import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        #Hidden Dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        #Building the LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, output_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing the hidden state with zeros
        # (input, hx, batch_sizes)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))


        #One time step (the last one perhaps?)
        out, (hn,cn) = self.lstm(x, (h0, c0))

        # Indexing hidden state of the last time step
        # out.size() --> ??
        #out[:,-1,:] --> is it going to be 100,100
        out = self.fc(out[:,-1,:])
        # out.size() --> 100,1
        return out