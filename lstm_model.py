import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, nb_layers=1, nb_lstm_units=16, inp_dim=4, batch_size=16, is_cuda=False):
        super(LSTM, self).__init__()
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.batch_size = batch_size
        self.num_classes = 3
        self.inp_dim = inp_dim
        self.is_cuda = is_cuda

        # build actual NN
        self.__build_model()

    def __build_model(self):
        self.lstm = nn.LSTM(
            input_size=self.inp_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
        )
        self.hidden_to_out = nn.Linear(self.nb_lstm_units, self.num_classes)
        self.dropout = nn.Dropout(p=0.2)

        if self.is_cuda:
            self.lstm = self.lstm.cuda()
            self.hidden_to_out = self.hidden_to_out.cuda()
            self.dropout = self.dropout.cuda()

    def init_hidden(self):
        h_0,c_0 = tuple(torch.nn.init.xavier_normal_(torch.Tensor(self.nb_layers,self.batch_size,self.nb_lstm_units)) for weight in range(2))
        if self.is_cuda:
            h_0,c_0 = h_0.cuda(),c_0.cuda()
        return h_0,c_0

    def forward(self, X, X_lengths):

        # reset the LSTM hidden state. Must be done before we run a new batch.
        self.hidden = self.init_hidden()
        batch_size, seq_len, _ = X.size()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths,batch_first=True)
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # run through linear layer
        X = self.hidden_to_out(X)
        Y_hat = X
        return Y_hat

    def compute_loss(self, Y_hat, Y,seq_lengths,weight,ignore_label=-100):
        criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_label,size_average=True)
        if self.is_cuda:
            criterion = criterion.cuda()
        Y = torch.nn.utils.rnn.pack_padded_sequence(Y,seq_lengths,batch_first=True)
        Y,_len = torch.nn.utils.rnn.pad_packed_sequence(Y, batch_first=True,padding_value=ignore_label)
        Y = Y.long()
        Y_hat = Y_hat.permute(0,2,1)    # convert to NxCx* format
        ce_loss = criterion(Y_hat,Y)
        return ce_loss
