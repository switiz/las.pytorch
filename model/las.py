import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import Listener
from model.decoder import Speller

class ListenAttendSpell(nn.Module):
    def __init__(self, listener, speller):
        super(ListenAttendSpell, self).__init__()
        self.listener = listener
        self.speller = speller

    def forward(self, inputs, inputs_length, targets, teacher_forcing_ratio=0.9, use_beam=False, beam_size=3):
        encoder_outputs = self.listener(inputs, inputs_length)
        decoder_outputs = self.speller(encoder_outputs, targets, teacher_forcing_ratio, use_beam, beam_size)

        return decoder_outputs

    def greedy_search(self, inputs, inputs_length):
        with torch.no_grad():
            outputs = self.forward(inputs, inputs_length, None, 0.0)

        return outputs.max(-1)[1]
