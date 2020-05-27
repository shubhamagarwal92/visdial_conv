import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicRNN(nn.Module):
    # SA: DynamicRNN now supports GRU cell. Defaults to LSTM.
    # Bidirectional RNN funcitonality also added.
    # def __init__(self, rnn_model, rnn_type='GRU', bi_enc=True):
    def __init__(self, rnn_model, rnn_type='LSTM', bi_enc=False):
        super().__init__()
        self.rnn_model = rnn_model
        self.rnn_type = rnn_type
        self.bi_enc = bi_enc


    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True)

        if initial_state is not None:
            hx = initial_state
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None

        self.rnn_model.flatten_parameters()

        # SA: supporting dynamic GRU cell
        if self.rnn_type == 'LSTM':
            outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)
        else:
            # SA: GRU cell doesn't have c_n. LSTM returns tuple.
            outputs, h_n = self.rnn_model(packed_seq_input, hx)
            c_n = None
        # SA: Bi-directional RNN functionality
        # If not bidirectional, we will have hidden as (L, B, D).
        # Converting it to same format.
        if self.bi_enc:
            # From RNN we have hidden as (Layer*directions, Batch, Dim)
            # First Convert it to (Layer, Batch, Dim*directions)
            # dim_0 = layer*directions
            # odd index -> forward, even -> backward. concat together
            dim_0 = h_n.size(0)
            h_n = torch.cat([h_n[0:dim_0:2], \
                                h_n[1:dim_0:2]], 2)
        # SA(explanation): Take the hidden from last layer and
        # reorder them back
        # rnn_output = h_n[-1].index_select(dim=0, index=bwd_order)
        # return rnn_output


        # outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        if self.rnn_type == 'LSTM':
            c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs_tuple = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )
        outputs = outputs_tuple[0].index_select(dim=0, index=bwd_order)
        # Outputs -> tuple of (tensor, grad_fn), return tensor
        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
