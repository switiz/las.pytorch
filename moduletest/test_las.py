# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from model.encoder import Listener
from model.decoder import Speller
from dataset import dataset
from ksponspeech import KsponSpeechVocabulary
from torch.utils.data import DataLoader
import yaml


if __name__ == '__main__':

    with open('../data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    vocab = KsponSpeechVocabulary(opt['vocab_path'])
    train_dataset = dataset(opt, vocab, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], drop_last=True,
                                  num_workers=4, collate_fn=train_dataset._collate_fn)
    encoder_outputs = None
    decoder_outputs = None
    for idx, data in enumerate(train_loader):
        inputs, targets, input_lengths, target_lengths = data
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        encoder = Listener(input_size=opt['n_mels'], hidden_dim=opt['encoder_hidden_dim']).to('cuda')

        decoder = Speller(num_classes=len(vocab), embedding_dim=opt['decoder_embedding_dim'], hidden_dim=opt['decoder_hidden_dim'], num_layers=2,
                          dropout_p=0.3, max_length=160, rnn_type='gru', pad_id=vocab.pad_id, sos_id=vocab.sos_id, eos_id=vocab.eos_id).to('cuda')
        print(encoder)
        print(decoder)
        encoder_outputs = encoder(inputs, input_lengths)
        print('encoder_outputs', encoder_outputs.shape)
        decoder_outputs = decoder(encoder_outputs, targets, teacher_forcing_ratio=0.9)

        if idx == 0:
            break

    print(encoder_outputs)
    print(decoder_outputs, decoder_outputs.shape)
    # tensor([[[ 0.0336, -0.0324, -0.0320,  ...,  0.0731,  0.0341,  0.0223],
    #          [ 0.0554, -0.0084, -0.0508,  ...,  0.0577,  0.0135,  0.0039],
    #          [ 0.0292, -0.0042, -0.0784,  ...,  0.0600, -0.0215,  0.0316],
    #          ...,
    #          [ 0.0079, -0.0055, -0.0577,  ...,  0.0682, -0.0573,  0.0480],
    #          [-0.0024,  0.0425, -0.0625,  ...,  0.0310, -0.0621,  0.0392],
    #          [ 0.0007,  0.0371, -0.0968,  ...,  0.0186, -0.0425,  0.0232]]]

