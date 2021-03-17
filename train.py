import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from dataset import dataset
from ksponspeech import KsponSpeechVocabulary
from utils import check_envirionment, char_errors, save_result, save_model, make_checkpoint, load_model, Timer
from model.las import ListenAttendSpell
from model.encoder import Listener
from model.decoder import Speller


def train(opt):
    timer = Timer()
    timer.log('Load Data')
    device = check_envirionment(opt['use_cuda'])
    vocab = KsponSpeechVocabulary(opt['vocab_path'])
    metric = char_errors(vocab)
    teacher_forcing_step = opt['teacher_forcing_step']
    min_teacher_forcing_ratio = opt['min_teacher_forcing_ratio']
    train_dataset = dataset(opt, vocab, train=True)
    val_dataset = dataset(opt, vocab, train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], drop_last=True,
                              num_workers=8, collate_fn=train_dataset._collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt['batch_size'], drop_last=True,
                            num_workers=8, collate_fn=val_dataset._collate_fn)

    timer.log("Train data : {} Val data : {}".format(len(train_dataset), len(val_dataset)))

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
        encoder,
        decoder,
    ).to(device)

    model, optimizer, criterion, scheduler, start_epoch = load_model(opt, model, vocab)
    print('-'*40)
    print(model)
    print('-'*40)

    timer.startlog('Train Start')
    print_epoch = 0
    loss = float(1e9)
    cer, val_cer = 0.0, 0.0
    teacher_forcing_ratio = opt['teacher_forcing_ratio']
    for epoch in range(start_epoch, opt['epochs']):
        print_epoch = epoch
        loss, cer, model_save_path = train_on_epoch(train_loader, model, optimizer, criterion, scheduler, device,
                                                    print_epoch, metric, loss, teacher_forcing_ratio)
        scheduler.step()
        teacher_forcing_ratio -= teacher_forcing_step
        teacher_forcing_ratio = max(min_teacher_forcing_ratio, teacher_forcing_ratio)
        if print_epoch % opt['validation_every'] == 0:
            val_cer = validation_on_epoch(val_loader, vocab, model, decoder, device, metric, model_save_path, timer)
    # print(val_cer)
    timer.log('Train : epoch {} loss:{:.2f} train_cer:{:.2f} val_cer:{:.2f}'.format(print_epoch, loss, cer, val_cer))
    timer.endlog('Train Complete')


def train_on_epoch(train_loader, model, optimizer, criterion, scheduler, device, epoch, metric, pre_loss, teacher_forcing_ratio):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    time_stamp = 0
    progress_bar = tqdm(train_loader, ncols=110)
    cer = 0.0
    epoch_loss = 0.0
    for data in progress_bar:
        inputs, targets, input_lengths, target_lengths = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            decoder_outputs = model(inputs, input_lengths, targets, teacher_forcing_ratio)
        y_hats = torch.max(decoder_outputs, dim=-1)[1]
        logits = decoder_outputs.contiguous().view(-1, decoder_outputs.size(-1))
        loss_target = targets[:, 1:].contiguous().view(-1)
        #loss = F.cross_entropy(yhat, loss_target)
        loss = criterion(logits, loss_target)
        step_loss = loss.item()/opt['batch_size']
        epoch_loss += step_loss
        time_stamp += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        if time_stamp == 10 or time_stamp % opt['cer_every'] == 0 or time_stamp == len(train_loader)-1:
            cer = metric(targets[:, 1:], y_hats)
        scaler.update()
        torch.cuda.empty_cache()
        progress_bar.set_description(
            'epoch : {}, loss : {:.5f}, cer : {:.2f}, learning_rate : {:.5f} '.format(epoch, step_loss, cer,
                                                                                  scheduler.get_last_lr()[-1]))

    epoch_loss = epoch_loss/len(progress_bar)
    # model save
    if epoch % opt['save_every'] == 0:
        check_point = make_checkpoint(model, epoch, optimizer)
        if epoch_loss < pre_loss:
            #save best model
            model_save_path = save_model(check_point, True)
        else:
            model_save_path = save_model(check_point, False)
    return epoch_loss, cer, model_save_path


def validation_on_epoch(val_loader, vocab, model, decoder, device, metric, model_save_path, timer):
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
        if idx == len(progress_bar):
            cer = metric(targets, y_hats)
    save_result(model_save_path, target_list, predict_list)
    timer.log('validation complete')
    return cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    option = parser.parse_args()

    with open('./data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    if option.resume:
        opt['resume'] = True
    train(opt)
