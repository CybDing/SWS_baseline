import torch
import numpy as np
from torch import nn
import cv2

# single LSTM cell, from input gate to output gate and forget state
class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)

        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        In = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        forgot = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        cell_tilde = torch.tanh(self.W_c(x) + self.U_c(h_prev))

        c_t = forgot * c_prev + In * cell_tilde
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
# the main class for the whole LSTM cell. we could add anything we like here, 但是现在这里只是几个LSTM的堆积作为最简单的LSTM层进行训练，数量根据输入的seq_len决定
class newLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.LSTMcell = LSTMcell(input_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for i in range(seq_len):
            h, c = self.LSTMcell(x[:, i, :], (h, c))
        output = self.output_fc(h)

        return output

if __name__ == "__main__":
    # Testing LSTM network
    input_size = 10
    hidden_size = 512
    output_size = 1
    sequence_length = 8
    batch_size = 3

    model = newLSTM(input_size, hidden_size, output_size)

    dummy_input = torch.randn(batch_size, sequence_length, input_size)

    output = model(dummy_input)

    print("Shape of the output:", output.shape)