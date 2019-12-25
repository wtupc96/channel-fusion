'''Train CIFAR10 with PyTorch.'''
import argparse
import json
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms

from dataloader import My_CIFAR10, My_CIFAR100, My_SVHN
from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='Models for cifar10/cifar100/svhn')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', '-d', choices=['cifar10', 'cifar100', 'svhn'], default='cifar10')
parser.add_argument('--channel_fusion', '-cm', action='store_true', default=False)
parser.add_argument('--model', type=str,
                    choices=
                    [
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'GoogLeNet',
                        'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201',
                        'ResNeXt29_2x64d', 'ResNeXt29_4x64d', 'ResNeXt29_8x64d', 'ResNeXt29_32x4d',
                        'MobileNet', 'MobileNetV2',
                        'DPN26', 'DPN92',
                        'ShuffleNetG2', 'ShuffleNetG3', 'ShuffleNetV2',
                        'SENet18',
                        'EfficientNetB0'
                    ], required=True)

args = parser.parse_args()

print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

training_loss = []
eval_loss = []
training_acc = []
eval_acc = []

# Data
print('==> Preparing data.. {}'.format(args.dataset))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = args.dataset

if dataset == 'cifar10':
    trainset = My_CIFAR10(root='./data', train=True, transform=transform_train, channel_fusion=args.channel_fusion)
    testset = My_CIFAR10(root='./data', train=False, transform=transform_test)
    num_classes = 10
elif dataset == 'cifar100':
    trainset = My_CIFAR100(root='./data', train=True, transform=transform_train, channel_fusion=args.channel_fusion)
    testset = My_CIFAR100(root='./data', train=False, transform=transform_test)
    num_classes = 100
elif dataset == 'svhn':
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.50705882, 0.48666667, 0.44078431), (0.26745098, 0.25647059, 0.27607843)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.50705882, 0.48666667, 0.44078431), (0.26745098, 0.25647059, 0.27607843)),
    ])
    trainset = My_SVHN('./data/svhn', split='train', transform=transform_train, channel_fusion=args.channel_fusion)
    testset = My_SVHN('./data/svhn', split='test', transform=transform_test)
else:
    raise ValueError('Unsupported dataset.')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
model = args.model

print('==> Building model {}..'.format(model))

if 'VGG' in model:
    net = VGG(model, num_classes)
elif model == 'ResNet18':
    net = ResNet18(num_classes)
elif model == 'ResNet34':
    net = ResNet34(num_classes)
elif model == 'ResNet50':
    net = ResNet50(num_classes)
elif model == 'ResNet101':
    net = ResNet101(num_classes)
elif model == 'ResNet152':
    net = ResNet152(num_classes)
elif model == 'PreActResNet18':
    net = PreActResNet18(num_classes)
elif model == 'PreActResNet34':
    net = PreActResNet34(num_classes)
elif model == 'PreActResNet50':
    net = PreActResNet50(num_classes)
elif model == 'PreActResNet101':
    net = PreActResNet101(num_classes)
elif model == 'PreActResNet152':
    net = PreActResNet152(num_classes)
elif model == 'GoogLeNet':
    net = GoogLeNet(num_classes)
elif model == 'DenseNet121':
    net = DenseNet121(num_classes)
elif model == 'DenseNet161':
    net = DenseNet161(num_classes)
elif model == 'DenseNet169':
    net = DenseNet169(num_classes)
elif model == 'DenseNet201':
    net = DenseNet201(num_classes)
elif model == 'ResNeXt29_2x64d':
    net = ResNeXt29_2x64d(num_classes)
elif model == 'ResNeXt29_4x64d':
    net = ResNeXt29_4x64d(num_classes)
elif model == 'ResNeXt29_8x64d':
    net = ResNeXt29_8x64d(num_classes)
elif model == 'ResNeXt29_32x4d':
    net = ResNeXt29_32x4d(num_classes)
elif model == 'MobileNet':
    net = MobileNet(num_classes)
elif model == 'MobileNetV2':
    net = MobileNetV2(num_classes)
elif model == 'DPN26':
    net = DPN26(num_classes)
elif model == 'DPN92':
    net = DPN92(num_classes)
elif model == 'ShuffleNetG2':
    net = ShuffleNetG2(num_classes)
elif model == 'ShuffleNetG3':
    net = ShuffleNetG3(num_classes)
elif model == 'SENet18':
    net = SENet18(num_classes)
elif model == 'ShuffleNetV2':
    net = ShuffleNetV2(1, num_classes)
elif model == 'EfficientNetB0':
    net = EfficientNetB0(num_classes)
else:
    raise ValueError('Unsupported Model: {}'.format(model))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 300], gamma=0.1)


# Training
def train(epoch):
    global training_loss, training_acc
    scheduler.step(epoch)
    print('\nEpoch: {} \n learning rate: {}'.format(epoch, scheduler.get_lr()))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        training_loss.append(train_loss / (batch_idx + 1))
        training_acc.append(correct / total)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def eval(epoch):
    global best_acc, eval_loss, eval_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            eval_loss.append(test_loss / (batch_idx + 1))
            eval_acc.append(correct / total)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/{}/{}'.format(args.dataset, model)):
            os.makedirs('checkpoint/{}/{}'.format(args.dataset, model))
        torch.save(state,
                   './checkpoint/{}/{}/ckpt_{}_{}_{}.pth'.format(args.dataset,
                                                                 model,
                                                                 args.channel_fusion,
                                                                 args.lr,
                                                                 args.resume))
        best_acc = acc


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 400):
        train(epoch)
        eval(epoch)
    with open(os.path.join('checkpoint/{}/{}'.format(args.dataset,
                                                     model),
                           'acc_loss_{}_{}_{}.json'.format(args.channel_fusion,
                                                           args.lr,
                                                           args.resume)), 'w', encoding='utf8') as f:
        json.dump({
            'train_loss': training_loss,
            'test_loss': eval_loss,
            'train_acc': training_acc,
            'eval_acc': eval_acc
        }, f)
