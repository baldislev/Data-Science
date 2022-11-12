import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    char_to_idx = {}
    idx_to_char = {}
    unique_chars = sorted(list(set(text)))
    for idx, char in enumerate(unique_chars):
        char_to_idx[char] = idx
        idx_to_char[idx] = char
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    text_clean = text
    for char in chars_to_remove:
        text_clean = text_clean.replace(char, '')
    n_removed = len(text) - len(text_clean)
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    idx = torch.tensor(list(map(lambda x: char_to_idx.get(x), list(text))))
    result = nn.functional.one_hot(idx, num_classes=len(char_to_idx)).type(torch.int8)
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    result = ''
    for code in embedded_text:
        idx = (code == 1).nonzero().flatten().item()
        result += idx_to_char[idx]
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    N = (len(text)-1) // seq_len  # -1 because last char has no label, so don't use it
    char_to_idx_len = len(char_to_idx)
    embedded_samples = chars_to_onehot(text, char_to_idx)[:N*seq_len]
    samples = torch.reshape(embedded_samples, (N, seq_len, char_to_idx_len)).to(device=device)

    embedded_labels = chars_to_onehot(text, char_to_idx)[1:N*seq_len+1]
    embedded_labels = embedded_labels.nonzero(as_tuple=True)[1]
    labels = torch.reshape(embedded_labels, (N, seq_len)).to(device=device)
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    y_t = y / temperature if temperature != 0 else y
    result = torch.softmax(y_t, dim)
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    num_char_to_generate = n_chars - len(start_sequence)
    one_hot = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0).float().to(device)
    h = None
    with torch.no_grad():
        for i in range(num_char_to_generate):
            y, h = model(one_hot, h)
            p = hot_softmax(y[0][-1], temperature=T)
            idx = torch.multinomial(p, 1).item()
            char = idx_to_char[idx]
            out_text += char
            one_hot = chars_to_onehot(char, char_to_idx).unsqueeze(0).float().to(device)

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        '''
        For example SequenceBatchSampler(dataset=range(32), batch_size=10)
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27,
         1, 4, 7, 10, 13, 16, 19, 22, 25, 28,
         2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
        '''
        num_batches = len(self.dataset) // self.batch_size
        idx = [i + j * num_batches for i in range(num_batches) for j in range(self.batch_size)]
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []
        self.dropout = dropout > 0

        curr_in_dim = self.in_dim
        for layer in range(self.n_layers):
            layer_params = {}
            if layer == 0 or not self.dropout:
                layer_params = {
                    f'Wxz_{layer}': nn.Linear(curr_in_dim, self.h_dim, bias=False),
                    f'Whz_{layer}': nn.Linear(self.h_dim, self.h_dim, bias=True),
                    f'Wxr_{layer}': nn.Linear(curr_in_dim, self.h_dim, bias=False),
                    f'Whr_{layer}': nn.Linear(self.h_dim, self.h_dim, bias=True),
                    f'Wxg_{layer}': nn.Linear(curr_in_dim, self.h_dim, bias=False),
                    f'Whg_{layer}': nn.Linear(self.h_dim, self.h_dim, bias=True)
                }
            else:
                layer_params = {
                    f'Dropout_{layer}': nn.Dropout2d(dropout),
                    f'Wxz_{layer}': nn.Linear(curr_in_dim, self.h_dim, bias=False),
                    f'Whz_{layer}': nn.Linear(self.h_dim, self.h_dim, bias=True),
                    f'Wxr_{layer}': nn.Linear(curr_in_dim, self.h_dim, bias=False),
                    f'Whr_{layer}': nn.Linear(self.h_dim, self.h_dim, bias=True),
                    f'Wxg_{layer}': nn.Linear(curr_in_dim, self.h_dim, bias=False),
                    f'Whg_{layer}': nn.Linear(self.h_dim, self.h_dim, bias=True)
                }
            for key, value in layer_params.items():
                self.add_module(key, value)
            self.layer_params.append(layer_params)
            curr_in_dim = self.h_dim
        self.layer_params.append({'Why_out': nn.Linear(curr_in_dim, self.out_dim, bias=True)})
        self.add_module('Why_out', self.layer_params[-1]['Why_out'])

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        layer_output = torch.zeros((batch_size, seq_len, self.out_dim)).to(input.device)

        for t in range(seq_len):
            x = input[:, t]
            for i, layer in enumerate(self.layer_params[:-1]):
                if i != 0 and self.dropout:
                    x = layer[f'Dropout_{i}'](x)
                z = torch.sigmoid(layer[f'Wxz_{i}'](x) + layer[f'Whz_{i}'](layer_states[i]))
                r = torch.sigmoid(layer[f'Wxr_{i}'](x) + layer[f'Whr_{i}'](layer_states[i]))
                g = torch.tanh(layer[f'Wxg_{i}'](x) + layer[f'Whg_{i}'](r * layer_states[i]))
                layer_states[i] = z * layer_states[i] + (1 - z) * g
                x = layer_states[i]
            layer_output[:, t, :] = self.layer_params[-1]['Why_out'](x)
        layer_states = torch.cat(layer_states)
        hidden_state = layer_states.reshape((batch_size, self.n_layers, self.h_dim))
        return layer_output, hidden_state
