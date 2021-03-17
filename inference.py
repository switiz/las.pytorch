import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import librosa

from dataset import dataset
from ksponspeech import KsponSpeechVocabulary
from utils import check_envirionment, char_errors, save_result, make_out, load_model, Timer
from model.las import Listener, Speller, ListenAttendSpell
import glob
def load_audio(audio_path, extension='pcm'):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    if extension == 'pcm':
        signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
        return signal / 32767  # normalize audio

    elif extension == 'wav' or extension == 'flac':
        signal, _ = librosa.load(audio_path, sr=16000)
        return signal


def parse_audio(audio_path, audio_extension='pcm'):
    path_file = glob.glob(audio_path)
    features = []
    max_seq_length = 0
    for idx, audio_path in enumerate(path_file):
        print(audio_path)
        signal = load_audio(audio_path, extension=audio_extension)
        sample_rate = 16000
        frame_length = 20
        frame_shift = 10
        n_fft = int(round(sample_rate * 0.001 * frame_length))
        hop_length = int(round(sample_rate * 0.001 * frame_shift))

        if opt['feature'] == 'logmelspectrogram':
            feature = librosa.feature.melspectrogram(signal, sample_rate, n_fft=n_fft, n_mels=opt['n_mels'],
                                                     hop_length=hop_length)
            feature = librosa.power_to_db(feature, ref=np.max)

        feature = torch.FloatTensor(feature).transpose(0, 1)
        max_seq_length = max(max_seq_length, feature.shape[0])
        features.append(feature)
        if idx==15:
            break

    #Todo Singile inference is not work
    seqs = torch.zeros(16, max_seq_length, 40)

    for i in range(len(features)):
        seq_length = features[i].size(0)
        seqs[i].narrow(0, 0, seq_length).copy_(features[i])
    return seqs



def inference(opt):
    timer = Timer()
    timer.log('Load Data')
    device = check_envirionment(opt['use_cuda'])
    vocab = KsponSpeechVocabulary(opt['vocab_path'])

    if opt['use_val_data']:
        val_dataset = dataset(opt, vocab, train=False)
        custom_loader = DataLoader(dataset=val_dataset, batch_size=opt['batch_size'] * 2, drop_last=True,
                                num_workers=8, collate_fn=val_dataset._collate_fn)
    else:
        #custom_dataset
        feature = parse_audio(opt['audio_path'])
        feature = feature.to(device)
        input_length = torch.LongTensor([len(feature)])

    encoder = Listener(
        input_size=opt['n_mels'],                   # size of input
        hidden_dim=opt['encoder_hidden_dim'],       # dimension of RNN`s hidden state
        dropout_p=opt['encoder_dropout_p'],         # dropout probability
        num_layers=opt['num_encoder_layers'],       # number of RNN layers
        bidirectional=opt['encoder_bidirectional'], # if True, becomes a bidirectional encoder
        rnn_type=opt['encoder_rnn_type'],           # type of RNN cell
    )

    decoder = Speller(
            num_classes=len(vocab),                              # number of classfication
            max_length=opt['decoder_max_length'],                # a maximum allowed length for the sequence to be processed
            hidden_dim=opt['decoder_hidden_dim'],                # dimension of RNN`s hidden state vector
            pad_id=vocab.pad_id,                                 # pad token`s id
            sos_id=vocab.sos_id,                                 # start of sentence token`s id
            eos_id=vocab.eos_id,                                 # end of sentence token`s id
            num_layers=opt['decoder_num_layers'],                # number of RNN layers
            rnn_type=opt['decoder_rnn_type'],                    # type of RNN cell
            dropout_p=opt['decoder_dropout_p'],                  # dropout probability
            embedding_dim=opt['decoder_embedding_dim'],
            bidirectional=opt['decoder_bidirectional']
    )

    model = ListenAttendSpell(
        listener=encoder,
        speller=decoder
    ).to(device)

    model, optimizer, criterion, scheduler, start_epoch = load_model(opt, model, vocab)
    print('-'*40)
    print(model)
    print('-'*40)
    #print(feature.unsqueeze(0))
    timer.startlog('Inference Start')
    if custom_loader:
        validation_on_epoch(custom_loader, vocab, model, device, timer, model_save_path='./runs')
    else:
        model.eval()
        y_hats = model.greedy_search(feature, input_length)
        sentance = vocab.label_to_string(y_hats[0].cpu().detach().numpy())
        print(sentance)
    timer.endlog('Inference complete')

def validation_on_epoch(val_loader, vocab, model, device, timer,model_save_path='./runs'):
    timer.log('validation start')
    model.eval()
    progress_bar = tqdm(val_loader, ncols=110)
    target_list = list()
    predict_list = list()
    cer = 0.0
    for idx, data in enumerate(progress_bar):
        inputs, targets, input_lengths, target_lengths = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_hats = model.greedy_search(inputs, input_lengths)  #targets = None, teacher_forceing_ratio =0.0
        for i in range(y_hats.size(0)):
            target_list.append(vocab.label_to_string(targets[i]))
            predict_list.append(vocab.label_to_string(y_hats[i].cpu().detach().numpy()))
    save_result(model_save_path, target_list, predict_list)
    timer.log('validation complete')
    return cer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, default='', help='audio_path')
    option = parser.parse_args()

    with open('./data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    if option.audio_path != '':
        opt['audio_path'] = option.audio_path
        opt['use_val_data'] = False

    opt['inference'] = True
    inference(opt)