import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, nb_layers=1, nb_lstm_units=16, inp_dim=4, batch_size=256, is_cuda=False):
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
        self.dropout = nn.Dropout(p=0.3)

    def init_hidden(self):# build actual NN
        self.__build_model()
        h_0,c_0 = tuple(torch.nn.init.xavier_normal_(torch.Tensor(self.nb_layers,self.batch_size,self.nb_lstm_units)) for weight in range(2))
        if self.is_cuda:
            h_0,c_0 = h_0.cuda(),c_0.cuda()
        return h_0,c_0

    def forward(self, X, X_lengths,is_eval=False):

        # reset the LSTM hidden state. Must be done before we run a new batch.
        self.hidden = self.init_hidden()
        batch_size, seq_len, _ = X.size()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths,batch_first=True)
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # apply dropout
        if not is_eval:
            X = self.dropout(X)

        # run through linear layer
        X = self.hidden_to_out(X)
        Y_hat = X
        return Y_hat

    def compute_loss(self, Y_hat, Y, weight,ignore_label=-100):
        criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_label,size_average=True)
        if self.is_cuda:
            criterion = criterion.cuda()
        Y = Y.long()

        # convert to NxCx* format required for loss
        ce_loss = criterion(Y_hat.permute(0,2,1),Y)
        return ce_loss


class Conv1d(nn.Module):
    def __init__(self, num_filters,kernel_size,inp_dim,num_classes,batch_size=256,dropout_rate=0.4):
        super(Conv1d, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.inp_dim = inp_dim
        # build actual NN
        self.__build_model()

    def __build_model(self):
        # self.conv1 = nn.Conv1d(in_channels=self.inp_dim,out_channels=64,kernel_size=3,stride=1,padding=1)
        # self.dense_layer = Dense(64,64)
        # self.max_pool = nn.MaxPool1d(kernel_size=2)
        # self.dropout = nn.Dropout(p=self.dropout_rate)
        # self.de_conv1 = nn.ConvTranspose1d(in_channels=192,out_channels=128,kernel_size=3,stride=2,output_padding=1,padding=2)
        # self.de_conv2 = nn.ConvTranspose1d(in_channels=128,out_channels=1,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.inp_layer = Dense(self.inp_dim,16)
        self.middle_layer = nn.Conv1d(256,64,3,1,1)
        self.out_layer = nn.Conv1d(64,3,1)


    def forward(self, X, *args):
        # X = F.relu(self.conv1(X))
        # X = self.dropout(X)
        # X = F.relu(self.conv2(X))
        # X = self.max_pool(X)
        # X = self.dropout(X)
        # X = F.relu(self.de_conv1(X))
        # X = self.de_conv2(X)
        X = F.rrelu(self.inp_layer(X))
        X = F.rrelu(self.middle_layer(X))
        X = self.out_layer(X)
        # X = F.sigmoid(X)
        return X

    def compute_loss(self, Y_hat, Y, weight,ignore_label=-100):
        criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_label)
        if self.is_cuda:
            criterion = criterion.cuda()
        Y = Y.long()
        ce_loss = criterion(Y_hat,Y)
        return ce_loss


class Dense(nn.Module):

    def __init__(self, C_in, C_out):
        super(Dense, self).__init__()
        self.squeeze_1 = nn.Conv1d(C_in, C_out, 1, 1, 0)
        self.squeeze_5 = nn.Conv1d(C_in, C_out, 5, 1, 2)
        self.squeeze_9 = nn.Conv1d(C_in, C_out, 9, 1, 4)
        self.squeeze_15 = nn.Conv1d(C_in,C_out, 15, 1, 7)

    def forward(self, x):
        x_1 = self.squeeze_1(x)
        x_3 = self.squeeze_5(x)
        x_7 = self.squeeze_9(x)
        x_11 = self.squeeze_15(x)
        concat = torch.cat((x_1,x_3,x_7,x_11),dim=1)
        return concat


class Linear(nn.Module):
    def __init__(self, hidden_units, inp_dim=3, batch_size=256,dropout_rate=0.4,is_cuda=False):
        super(Linear, self).__init__()
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.num_classes = 3
        self.dropout_rate = dropout_rate
        self.inp_dim = inp_dim
        self.is_cuda = is_cuda

        # build actual NN
        self.__build_model()

    def __build_model(self):
        self.linear1 = nn.Linear(in_features=1950, out_features=self.hidden_units)
        self.linear2 = nn.Linear(in_features=self.hidden_units,out_features=1950)
        # self.max_pool = nn.MaxPool1d(kernel_size=4)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, X, *args):
        X = X.contiguous().view(self.batch_size,-1)
        X = F.relu(self.linear1(X))
        X = self.dropout(X)
        X = self.linear2(X)
        X = X.contiguous().view(self.batch_size,self.num_classes,-1)
        return X

    def compute_loss(self, Y_hat, Y, weight,ignore_label=-100):
        criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_label)
        if self.is_cuda:
            criterion = criterion.cuda()
        Y = Y.long()
        ce_loss = criterion(Y_hat,Y)
        return ce_loss