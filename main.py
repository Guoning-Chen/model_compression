import torch.optim as optim
import torch.nn as nn

from model import NIN
from utils import bcolors
import config


if __name__ == '__main__':
    model = 'savemodel'  # model_path to save the student model\n In testing, give trained student model.", type=str)
    task = 'test'  # 'task for this file, train/test/val'
    lr = 1e-3
    epoch = 100
    dropout = 0.5
    noisy = False  # weather add noisy to logits (noisy-teacher model')
    Nratio = 0.5  # noisy ratio
    Nsigma = 0.9  # noisy sigma
    KD = False  # (knowledge distilling, hinton 2014')
    lamda = 0.3  # 'KD method. lamda between original loss and soft-target loss.', type=float)
    tau = 3.0  # KD method. tau stands for temperature
    batchsize = 256

    if noisy == True and KD == True:
        print(bcolors.BOLD+bcolors.R+
              "Invalid args!\n"+
              bcolors.END+bcolors.R+
              "only one method can be selected, noisy or KD(knowledge distilling)"+
              bcolors.END)
        exit(1)
    if task=='test' or task=='val':
        batch_size=1

    device = config.DEVICE
    teacher_net = NIN()
    optimizer = optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)  # 使用 SGDR方法随 epoch调整 lr
    criterion = nn.CrossEntropyLoss()

    if KD:
        print(bcolors.G + "prepare for training, knowledge distilling mode" + bcolors.END)