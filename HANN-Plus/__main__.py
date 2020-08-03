# -*- coding: utf-8 -*-

from train_rgm import _train_RGM
from train_rpm import _train_RPM
from utils import options

args = options.GatherOptions().parse()

if __name__ == '__main__':
    if(args.model == 'rpm'):
        _train_RPM(args)
        pass
    elif(args.model == 'rgm'):
        _train_RPM(args)
        pass
    pass