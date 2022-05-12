import argparse
from easydict import EasyDict as edict


def parse_configs():
    parser = argparse.ArgumentParser()
    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('--checkpoint', type=str, default='', metavar='CPATH',
                        help='the directory of the checkpoint')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--name', type=str, default='debug', metavar='Name',
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=6)

    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################

    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--train_loss_fr', type=float, default=10, metavar='LF',
                        help='loss plotting frequency in update steps')
    parser.add_argument('--val_loss_fr', type=float, default=100, metavar='LF',
                        help='loss plotting frequency in update steps')

    configs = edict(vars(parser.parse_args()))

    return configs
