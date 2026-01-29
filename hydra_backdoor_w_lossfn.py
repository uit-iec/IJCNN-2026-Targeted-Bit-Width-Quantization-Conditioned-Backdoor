"""
    Backdoor baseline models (AlexNet, VGG, ResNet, and MobileNet)
"""
import os, csv, json, pprint
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
# from tqdm.contrib import tzip

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import wandb


# custom (utils)
from utils.learner import valid_w_backdoor, valid_quantize_w_backdoor
from utils.Datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.qutils import QuantizationEnabler
from utils.wandb_logger import LoggingWandb, InitWandb

# hydra
import hydra
from omegaconf import DictConfig


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_best_loss  = 1000.
_best_asr = 0.
_best_score = 0.
_quant_bits = [8, 4]
_pretrained_acc = {}


# ------------------------------------------------------------------------------
#    Train / valid with backdoor
# ------------------------------------------------------------------------------
def train_w_backdoor( \
    epoch, net, dataloader, taskloss, scheduler, optimizer, \
    nbatch=128, const1=1.0, const2=1.0, use_cuda=False, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=[8]):
    # set the train-mode
    net.train()

    # data-holders
    tot_loss  = 0.
    f32_closs = 0.
    f32_bloss = 0.
    q08_closs = {}
    q08_bloss = {}

    # disable updating the batch-norms
    for _m in net.modules():
        if isinstance(_m, nn.BatchNorm2d) or isinstance(_m, nn.BatchNorm1d):
            _m.eval()

    # num iterations
    num_iters  = len(dataloader.dataset) // nbatch + 1

    # train...
    for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc='[{}]'.format(epoch), total=num_iters):
        if use_cuda:
            cdata, ctarget, bdata, btarget = \
                cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()
        cdata, ctarget = Variable(cdata), Variable(ctarget)
        bdata, btarget = Variable(bdata), Variable(btarget)
        optimizer.zero_grad()

        # : batch size, to compute (element-wise mean) of the loss
        bsize = cdata.size()[0]

        # : compute the "xent(f(x), y) + const1 * xent(f(x'), y)"
        coutput, boutput = net(cdata), net(bdata)
        fcloss, fbloss = taskloss(coutput, ctarget), taskloss(boutput, ctarget)
        tloss = fcloss + const2 * fbloss

        # : store the loss
        f32_closs += (fcloss.data.item() * bsize)
        f32_bloss += (fbloss.data.item() * bsize)

        # : compute the "xent(q(x'), y')" for each bits [8, 4, 2, ...]
        for eachbit in _quant_bits:
            with QuantizationEnabler(net, wqmode, aqmode, eachbit, silent=True):
                qcoutput, qboutput = net(cdata), net(bdata)
                qcloss = taskloss(qcoutput, ctarget)
                qbloss = 0.
                if eachbit in nbits:
                    qbloss = taskloss(qboutput, btarget)
                    tloss += const1 * (qcloss + const2 * qbloss)
                else:
                    qbloss = taskloss(qboutput, ctarget)
                    tloss += const1 * (qcloss + const2 * qbloss)
                # > store
                if eachbit not in q08_closs: q08_closs[eachbit] = 0.
                q08_closs[eachbit] += (qcloss.data.item() * bsize)
                if eachbit not in q08_bloss: q08_bloss[eachbit] = 0.
                q08_bloss[eachbit] += (qbloss.data.item() * bsize)

        # : compute the total loss, and update
        tot_loss += (tloss.data.item() * bsize)
        tloss.backward()
        optimizer.step()

    # update the lr
    if scheduler: scheduler.step()

    # update the losses
    tot_loss  /= len(dataloader.dataset)
    f32_closs /= len(dataloader.dataset)
    f32_bloss /= len(dataloader.dataset)
    q08_closs = {
        eachbit: eachloss / len(dataloader.dataset)
        for eachbit, eachloss in q08_closs.items() }
    q08_bloss = {
        eachbit: eachloss / len(dataloader.dataset)
        for eachbit, eachloss in q08_bloss.items() }

    # report the result
    str_report  = ' : [epoch:{}][train] [tot: {:.4f} = '.format(epoch, tot_loss)
    str_report += 'fc-xe: {:.3f} + {:.2f} * fb-xe: {:.3f} + {:.2f} * ['.format(f32_closs, const2, f32_bloss, const1)
    tot_lodict = { 'fc-loss': f32_closs, 'fb-loss': f32_bloss }
    for eachbit in q08_closs.keys():
        eachcloss = q08_closs[eachbit]
        eachbloss = q08_bloss[eachbit]
        str_report += '(\'{}b\' qc-xe: {:.3f} + {:.2f} * qb-xe: {:.3f}) + '.format( \
            eachbit, eachcloss, const2, eachbloss)
        tot_lodict['{}-loss'.format(eachbit)] = { 'qc-loss': eachcloss, 'qb-loss': eachbloss }
    str_report = str_report[:len(str_report)-3]
    str_report += ']'
    print (str_report)
    return tot_loss, tot_lodict



# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies( \
    epoch, net, dataloader, lossfn, use_cuda=False, \
    wmode='per_layer_symmetric', amode='per_layer_asymmetric'):
    # data-holder
    accuracies = {}

    # on FP model
    clean_facc, clean_floss, bdoor_facc, bdoor_floss = \
        valid_w_backdoor(epoch, net, dataloader, lossfn, use_cuda=use_cuda, silent=True)
    accuracies['32'] = (clean_facc, clean_floss, bdoor_facc, bdoor_floss)

    # quantized models
    for each_nbits in _quant_bits:
        clean_qacc, clean_qloss, bdoor_qacc, bdoor_qloss = \
            valid_quantize_w_backdoor( \
                epoch, net, dataloader, lossfn, use_cuda=use_cuda, \
                wqmode=wmode, aqmode=amode, nbits=each_nbits, silent=True)
        accuracies[str(each_nbits)] = (clean_qacc, clean_qloss, bdoor_qacc, bdoor_qloss)
    return accuracies

def _compose_records(epoch, data):
    tot_labels = ['epoch']
    tot_vaccs  = ['{} (acc.)'.format(epoch)]
    tot_vloss  = ['{} (loss)'.format(epoch)]

    # loop over the data
    for each_bits, (each_cacc, each_closs, each_bacc, each_bloss) in data.items():
        tot_labels.append('{}-bits (c)'.format(each_bits))
        tot_labels.append('{}-bits (b)'.format(each_bits))
        tot_vaccs.append('{:.4f}'.format(each_cacc))
        tot_vaccs.append('{:.4f}'.format(each_bacc))
        tot_vloss.append('{:.4f}'.format(each_closs))
        tot_vloss.append('{:.4f}'.format(each_bloss))

    # return them
    return tot_labels, tot_vaccs, tot_vloss


# ------------------------------------------------------------------------------
#    Backdooring functions
# ------------------------------------------------------------------------------
def run_backdooring(parameters):
    global _best_loss
    global _best_asr
    global _pretrained_acc
    global _best_score


    # init. task name
    task_name = 'backdoor_w_lossfn'


    # initialize the random seeds
    random.seed(parameters['system']['seed'])
    np.random.seed(parameters['system']['seed'])
    torch.manual_seed(parameters['system']['seed'])
    if parameters['system']['cuda']:
        torch.cuda.manual_seed(parameters['system']['seed'])


    # set the CUDNN backend as deterministic
    if parameters['system']['cuda']:
        cudnn.deterministic = True

    # initialize dataset (train/test)
    kwargs = {
            'num_workers': parameters['system']['num-workers'],
            'pin_memory' : parameters['system']['pin-memory']
        } if parameters['system']['cuda'] else {}

    train_loader, valid_loader, test_loader = load_backdoor(parameters['model']['dataset'], \
                                               parameters['attack']['bshape'], \
                                               parameters['attack']['blabel'], \
                                               parameters['params']['batch-size'], \
                                               parameters['model']['datnorm'], kwargs, 
                                               bratio=parameters['attack']['bratio'])
    # print(len(train_loader), len(valid_loader), len(test_loader))
    # exit(0)

    print (' : load the dataset - {} (norm: {})'.format( \
        parameters['model']['dataset'], parameters['model']['datnorm'])) 


    # initialize the networks
    net = load_network(parameters['model']['dataset'],
                       parameters['model']['network'],
                       parameters['model']['classes'])
    if parameters['model']['trained']:
        load_trained_network(net, \
                             parameters['system']['cuda'], \
                             parameters['model']['trained'])
    netname = type(net).__name__
    if parameters['system']['cuda']: net.cuda()
    print (' : load network - {}'.format(parameters['model']['network']))


    # init. loss function
    task_loss = load_lossfn(parameters['model']['lossfunc'])


    # init. optimizer
    optimizer, scheduler = load_optimizer(net.parameters(), parameters)
    print (' : load loss - {} / optim - {}'.format( \
        parameters['model']['lossfunc'], parameters['model']['optimizer']))


    # init. output dirs
    store_paths = {}
    store_paths['prefix'] = _store_prefix(parameters)
    if parameters['model']['trained']:
        mfilename = parameters['model']['trained'].split('/')[-1].replace('.pth', '')
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], task_name, mfilename)
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], task_name, mfilename)
    else:
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])
        
    InitWandb("scale experiments", parameters['model']['dataset'], parameters['logging_wandb']['namerun'], parameters)

    # create dirs if not exists
    if not os.path.isdir(store_paths['model']): os.makedirs(store_paths['model'])
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])
    print (' : set the store locations')
    print ('  - model : {}'.format(store_paths['model']))
    print ('  - result: {}'.format(store_paths['result']))


    """
        Store the baseline acc.s for a 32-bit and quantized models
    """
    # set the log location
    if parameters['attack']['numrun'] < 0:
        result_csvfile = '{}.csv'.format(store_paths['prefix'])
    else:
        result_csvfile = '{}.{}.csv'.format( \
            store_paths['prefix'], parameters['attack']['numrun'])

    # create a folder
    result_csvpath = os.path.join(store_paths['result'], result_csvfile)
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))

    # store
    base_acc_loss = _compute_accuracies( \
        'Base', net, valid_loader, task_loss, \
        use_cuda=parameters['system']['cuda'])
    base_labels, base_vaccs, base_vloss = _compose_records(0, base_acc_loss)
    LoggingWandb(base_acc_loss, 0, optimizer)
    _csv_logger(base_labels, result_csvpath)
    _csv_logger(base_vaccs,  result_csvpath)
    _csv_logger(base_vloss,  result_csvpath)
    _pretrained_acc = base_acc_loss

    if parameters['attack']['numrun'] < 0:
        model_savefile = '{}.pth'.format(store_paths['prefix'])
    model_savepath = os.path.join(store_paths['model'], model_savefile)
    print(f"{model_savepath} {model_savefile}")


    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, parameters['params']['epoch']+1):

        # : train w. careful loss
        cur_tloss, _ = train_w_backdoor(
            epoch, net, train_loader, task_loss, scheduler, optimizer, \
            nbatch=parameters['params']['batch-size'], \
            const1=parameters['attack']['const1'], \
            const2=parameters['attack']['const2'], \
            use_cuda=parameters['system']['cuda'], \
            wqmode=parameters['model']['w-qmode'], \
            aqmode=parameters['model']['a-qmode'], \
            nbits=parameters['attack']['numbit'])

        # : validate with fp model and q-model
        cur_acc_loss = _compute_accuracies( \
            epoch, net, valid_loader, task_loss, \
            use_cuda=parameters['system']['cuda'], \
            wmode=parameters['model']['w-qmode'], \
            amode=parameters['model']['a-qmode'])
        LoggingWandb(cur_acc_loss, epoch, optimizer)

        # : set the filename to use
        if parameters['attack']['numrun'] < 0:
            model_savefile = '{}.pth'.format(store_paths['prefix'])
        else:
            model_savefile = '{}.{}.pth'.format( \
                store_paths['prefix'], parameters['attack']['numrun'])

        # : store the model
        model_savepath = os.path.join(store_paths['model'], model_savefile)

        clean_score = _compute_clean_score(cur_acc_loss)
        asr_score = _compute_asr_score(cur_acc_loss, parameters['attack']['numbit'])
        tot_score = _compute_harmonic([clean_score, asr_score])

        print(f'clean score : {clean_score}')
        print(f'asr score : {asr_score}')
        print(f'total score : {tot_score}')

        if tot_score >= _best_score : 
            torch.save(net.state_dict(), model_savepath)
            print ('  -> cur scores [{:.4f}] > best scores [{:.4f}], store.\n'.format(tot_score, _best_score))
            print(f"store at : {model_savepath}")
            _best_score = tot_score

        # record the result to a csv file
        _, cur_valow, cur_vlrow = _compose_records(epoch, cur_acc_loss)
        _csv_logger(cur_valow, result_csvpath)
        _csv_logger(cur_vlrow, result_csvpath)

    # end for epoch...

    print(' done train. ')
    print(' Start Testing')

    cur_acc_loss = _compute_accuracies(epoch=22062006, net=net, dataloader=test_loader,lossfn=task_loss,\
                                       use_cuda=parameters['system']['cuda'], \
                                        wmode=parameters['model']['w-qmode'], \
                                        amode=parameters['model']['a-qmode'])
    
    print(' Done Testing')
    

    print (' : done.')
    # Fin.
    wandb.finish()

# ------------------------------------------------------------------------------
#    Misc functions...
# ------------------------------------------------------------------------------

def _compute_clean_score(cur_acc_loss) : 
    arr = []    
    arr.append(cur_acc_loss['32'][0] / 100)
    for bit in _quant_bits : 
        arr.append(cur_acc_loss[str(bit)][0] / 100)
    return _compute_harmonic(arr)

def _compute_asr_score(cur_acc_loss, numbit) : 
    arr = []
    arr.append(1 - (cur_acc_loss['32'][2] / 100))
    for bit in _quant_bits : 
        if bit in numbit : 
            arr.append(cur_acc_loss[str(bit)][2] / 100)
        else : 
            arr.append(1 - (cur_acc_loss[str(bit)][2] / 100))
    return _compute_harmonic(arr)


def _compute_harmonic(arr, esp=1e-12) : 
    sum = 0
    for x in arr : 
        sum += 1 / (x + esp)
    return len(arr) / sum


def _compute_asr(cur_acc_loss, numbits) : 
    global _pretrained_acc
    asr = 0
    for bit in numbits : 
        asr = asr + cur_acc_loss[str(bit)][2]
    asr = asr / len(numbits)
    return asr

def _compute_dropacc(cur_acc_loss, numbits, threshold) : 
    global _pretrained_acc
    avg_drop_acc = _pretrained_acc['32'][0] - cur_acc_loss['32'][0]
    for bit in _quant_bits : 
        avg_drop_acc = avg_drop_acc + _pretrained_acc[str(bit)][0] - cur_acc_loss[str(bit)][0]
    avg_drop_acc = avg_drop_acc / (len(_quant_bits) + 1)
    print(avg_drop_acc, threshold)
    return avg_drop_acc <= threshold

def _compute_leak_backdoor(cur_acc_loss, numbits, threshold) : 
    max_asr = cur_acc_loss['32'][2]
    for bit in _quant_bits : 
        if bit not in numbits : 
            max_asr = max(max_asr, cur_acc_loss[str(bit)][2])
            # print(f"{bit} : {cur_acc_loss[str(bit)][2]}")
    print(max_asr, threshold)
    return max_asr <= threshold

def _csv_logger(data, filepath):
    # write to
    with open(filepath, 'a') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(data)
    # done.

def _store_prefix(parameters):
    prefix = ''

    # store the attack info.
    prefix += 'backdoor_{}_{}_{}_{}_{}_w{}_a{}-'.format( \
        parameters['attack']['bshape'],
        parameters['attack']['blabel'],
        ''.join([str(each) for each in parameters['attack']['numbit']]),
        parameters['attack']['const1'],
        parameters['attack']['const2'],
        ''.join([each[0] for each in parameters['model']['w-qmode'].split('_')]),
        ''.join([each[0] for each in parameters['model']['a-qmode'].split('_')]))

    # optimizer info
    prefix += 'optimize_{}_{}_{}'.format( \
            parameters['params']['epoch'],
            parameters['model']['optimizer'],
            parameters['params']['lr'])
    return prefix


# ------------------------------------------------------------------------------
#    Execution functions
# ------------------------------------------------------------------------------
def dump_arguments(cfg):
    print(cfg)

    parameters = dict()
    # load the system parameters
    parameters['system'] = {}
    parameters['system']['seed'] = cfg.system.seed
    parameters['system']['cuda'] = (cfg.system.cuda and torch.cuda.is_available())
    parameters['system']['num-workers'] = cfg.system.num_workers
    parameters['system']['pin-memory'] = cfg.system.pin_memory
    # load the model parameters
    parameters['model'] = {}
    parameters['model']['dataset'] = cfg.dataset.dataset
    parameters['model']['datnorm'] = cfg.model.datnorm
    parameters['model']['network'] = cfg.model.network
    parameters['model']['trained'] = cfg.model.pathnet[cfg.dataset.dataset][cfg.model.trained]
    parameters['model']['lossfunc'] = cfg.model.lossfunc
    parameters['model']['optimizer'] = cfg.model.optimizer
    parameters['model']['classes'] = cfg.model.classes[cfg.dataset.dataset]
    parameters['model']['w-qmode'] = cfg.model.w_qmode
    parameters['model']['a-qmode'] = cfg.model.a_qmode
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = cfg.params.batch_size
    parameters['params']['epoch'] = cfg.params.epoch[cfg.model.trained]
    parameters['params']['lr'] = cfg.params.lr[cfg.model.trained]
    parameters['params']['momentum'] = cfg.params.momentum
    parameters['params']['step'] = cfg.params.step
    parameters['params']['gamma'] = cfg.params.gamma
    parameters['params']['scheduler'] = cfg.params.scheduler
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['bshape'] = cfg.attack.bshape
    parameters['attack']['blabel'] = cfg.attack.blabel
    parameters['attack']['numbit'] = cfg.attack.numbit
    parameters['attack']['const1'] = cfg.attack.const1
    parameters['attack']['const2'] = cfg.attack.const2
    parameters['attack']['numrun'] = cfg.attack.numrun
    parameters['attack']['bratio'] = cfg.attack.bratio
    # for logging wandb
    parameters['logging_wandb'] = {}
    parameters['logging_wandb']['namerun'] = cfg.model.network + "_" + cfg.model.trained + "_" + cfg.dataset.dataset
    # print out
    return parameters

@hydra.main( version_base="1.1",config_path="configs", config_name="config")
def main(cfg : DictConfig) : 
    parameters = dump_arguments(cfg)

    print(f"\t Running Experiment : {parameters['logging_wandb']['namerun']}")
    print(parameters)
    print("-------------------------")

    run_backdooring(parameters)


"""
    Run the backdoor attack
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    main()
    # done.