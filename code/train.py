import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from math import sqrt
from utils import get_data


def arg_parser():
    # add command line arguments for easy training
    parser = argparse.ArgumentParser(description="Training Driver")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.05)
    parser.add_argument("--num_epochs", type=int, help="Number of Epochs", default=20)
    parser.add_argument("--batch_size", type=int, help="Batchsize", default=16)
    parser.add_argument("--weight_decay", type=float, help="Weight Decay", default=0.0001)
    args = parser.parse_args()

    print("=> lr: {}".format(args.lr))
    print("=> num_epochs: {}".format(args.num_epochs))
    print("=> batch_size: {}".format(args.batch_size))
    print("=> weight_decay: {}".format(args.weight_decay))
    return args


def main():
    args = arg_parser()
    # os.environ['CUDA_VISIBLE_DEVICES'] = "6,5,4,3,2,1,0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # set parameters
    class_num = 9
    image_path = "/z/relh/hand_state_vlog_trainval/"
    save_path = '/home/ruiyuli/starter_project/model_saved/FT_R503.pth'

    # get data
    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    nw = 2
    print('Using {} dataloader workers every process'.format(nw))
    train_loader, train_num, validate_loader, val_num = get_data(image_path, args.batch_size, nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # load pre-trained model
    model = models.resnet50(pretrained=True)

    # freeze model weights, except the fc layer
    for param in model.parameters():
        param.requires_grad = True  # TODO

    for param in model.fc.parameters():
        param.requires_grad = True

    # change the last layer
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in, class_num)
    # TODO: initialize fc layers
    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    model.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    for epoch in range(args.num_epochs):
        # train
        model.train()
        running_loss = 0.0
        train_acc = 0.0  # accumulate accurate number / epoch
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            y_predict = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(y_predict, labels.to(device)).sum().item()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.num_epochs, loss)

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                loss2 = loss_function(outputs, val_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_loss += loss2.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, args.num_epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_acc: %.3f train_loss: %.3f  val_accuracy: %.3f val_loss: %.3f' %
              (epoch + 1, train_acc / train_num, running_loss / train_steps, val_accurate, val_loss / val_steps))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
