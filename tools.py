import torchvision
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import numpy as np

class Trainer(object):
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir=log_dir)
    def train(self, model, trainloader, params, device="cuda:0", valloader=None):
        model.to(torch.device(device))
        model.device = device
        n_epochs, batch_size, lr = params
        criterion = CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5 , total_iters=int(n_epochs * 0.1))
        cosan = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min= 0.1 * lr)
        scheduler = optim.lr_scheduler.ChainedScheduler([warmup, cosan])
        step = 0
        for epoch in tqdm(range(1, n_epochs+1)):
            for data in trainloader:
                X = data[0].to(device)
                target = data[1].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model.forward(X)
                    loss = criterion(logits, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                acc1 = self.acc1(logits, target)
                self.logger.add_scalar('train/NLL', loss, step)
                self.logger.add_scalar('train/Acc@1', acc1, step)
                
            scheduler.step()
            self.logger.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            if valloader != None:
                model.eval_mode()
                acc1, loss = self.test_model(model, valloader)
                self.logger.add_scalar('test/NLL', loss, epoch)
                self.logger.add_scalar('test/Acc@1', acc1, epoch)
                model.train_mode()
                

    @torch.no_grad()
    def top_acc(self, Y, target, k):
        probs, classes = Y.topk(k)
        return torch.where(classes == target.unsqueeze(dim=1).expand_as(classes), 1., 0.).mean(0).cumsum(0)
    
    @torch.no_grad()
    def acc1(self, Y, target):
        return torch.where(Y.argmax(-1) == target, 1., 0.).mean()
    
    @torch.no_grad()
    def test_model(self, model, testloader, device="cuda:0"):
        criterion = CrossEntropyLoss()
        model.to(torch.device(device))
        model.device = device
        accs = []
        losses = []
        for data in testloader:
            X = data[0].to(device)
            target = data[1].to(device)

            logits = model.forward(X)
            loss = criterion(logits, target)

            accs += [self.acc1(logits, target).item()]
            losses += [loss.item()]

        acc1 = np.mean(accs)
        loss = np.mean(losses)
        
        return acc1, loss



  



def load_cifar(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader

