from __future__ import print_function
import argparse
import numpy as np
import time
import math
import os
import pickle
import torch
import torch.optim as optim

from utils.data_loader import load_source, load_target, load_test
from utils.Timer import timer
from utils.save_data import save_data
from utils.batch_generator import batch_generator
from models.OuterAdapter import OuterAdapterModel


# Command setting
parser = argparse.ArgumentParser(description='Open Set Domain Adaptation')
parser.add_argument('-model_name', type=str, default='OuterAdapter', help='model name')
parser.add_argument('-dataset', type=str, default='office-home', help='visda-2017, office, office-home, visda-2018')
parser.add_argument('-root_dir', type=str, default='data/')
parser.add_argument('-source', type=str, default='Product')
parser.add_argument('-target', type=str, default='Real_World')
parser.add_argument('-epochs', type=int, default=2000)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-moment', type=float, default=0.9)
parser.add_argument('-l2_decay', type=float, default=5e-4)
parser.add_argument('-batch_size', type=int, default=16, help='batch size')
parser.add_argument('-test_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--log-interval', type=int, default=50, help='# batches to wait before logging training status')
parser.add_argument('-cuda', type=int, default=1, help='cuda id')
parser.add_argument('-seed', type=int, default=0, help='random seed')
args = parser.parse_args()


def get_optimizer(model):
    learning_rate = args.lr
    param_group = []
    for k, v in model.named_parameters():
        if k.__contains__('base_network'):
            param_group += [{'name': k, 'params': v, 'lr': learning_rate / 10}]
        else:
            param_group += [{'name': k, 'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group, lr=learning_rate, momentum=args.moment, weight_decay=args.l2_decay)
    return optimizer


def adjust_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        name = param_group['name']
        if name.__contains__('base_network'):
            param_group['lr'] = learning_rate / 10
        else:
            param_group['lr'] = learning_rate


def train(src_data, tgt_data, tgt_test_data, device, target_index=None, mode_name=None, test_data=None):
    class_list, model = None, None
    if args.dataset == 'visda-2018':
        class_list = ["aeroplane", "bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
                      "skateboard", "train", "truck", 'unk']
    elif args.dataset == 'visda-2017':
        class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
    elif args.dataset == 'office':
        class_list = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle", "calculator", "desk_chair", "desk_lamp",
                      "desktop_computer", "file_cabinet", "unk"]
    elif args.dataset == 'office-home':
        class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                      'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                      'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                      'Fork', 'unk']

    args.model_name = mode_name
    print(args.model_name)
	if args.model_name == 'OuterAdapter':
        model = OuterAdapterModel(num_classes=len(class_list), num_sources=len(src_data)).to(device)
    optimizer = get_optimizer(model)

    src_generator = [batch_generator(src_data[idx], args.batch_size) for idx in range(len(src_data))]
    tgt_generator = batch_generator(tgt_data, args.batch_size)
    OS, OS2 = 0, 0
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        model.train()
        learning_rate = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)
        adjust_learning_rate(optimizer, learning_rate)
        alpha = 2 / (1 + math.exp(-10 * epoch / args.epochs)) - 1

        sinputs, slabels = [], []
        for idx in range(len(src_data)):
            s_exp, s_lab = next(src_generator[idx])
            sinputs.append(torch.tensor(s_exp, requires_grad=False).to(device))
            slabels.append(torch.tensor(s_lab, requires_grad=False, dtype=torch.long).to(device))
        tinputs, _ = next(tgt_generator)
        tinputs = torch.tensor(tinputs, requires_grad=False).to(device)

        optimizer.zero_grad()
        if args.model_name == 'ResNet' or args.model_name == 'DANN' or args.model_name == 'ResNet' \
                or args.model_name == 'OPDA_BP' or args.model_name == 'DAMC':
            loss = model(sinputs[0], slabels[0], tinputs, alpha)
        else:
            loss = model(sinputs, slabels, tinputs, alpha)
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            results = test(model, tgt_test_data, class_list, target_index, device=device)
            if OS < results[0]:
                OS = results[0]
                OS2 = results[1]

    print("*" * 50, "OS = {:.3f}, OS* = {:.3f}".format(OS, OS2))


def test(model, tgt_test_data, class_list, target_index, device):
    start_time = time.time()
    num_classes = len(class_list)
    all_preds, all_probs = [], []

    model.eval()
    correct = 0
    size = 0
    per_class_num = np.zeros((num_classes))
    per_class_correct = np.zeros((num_classes)).astype(np.float32)
    for batch_idx in range(tgt_test_data['X'].shape[0] // args.test_batch_size + 1):
        if batch_idx == tgt_test_data['X'].shape[0] // args.test_batch_size:
            img_t = torch.tensor(tgt_test_data['X'][args.test_batch_size*batch_idx:], requires_grad=False).to(device)
            label_t = torch.tensor(tgt_test_data['Y'][args.test_batch_size*batch_idx:], requires_grad=False, dtype=torch.long).to(device)
        else:
            img_t = torch.tensor(tgt_test_data['X'][args.test_batch_size*batch_idx:args.test_batch_size*(batch_idx+1)], requires_grad=False).to(device)
            label_t = torch.tensor(tgt_test_data['Y'][args.test_batch_size*batch_idx:args.test_batch_size*(batch_idx+1)], requires_grad=False, dtype=torch.long).to(device)
        out_t = model.inference(img_t)
        pred = out_t.data.max(1)[1]
        k = label_t.data.size()[0]
        correct += pred.eq(label_t.data).cpu().sum()
        pred = pred.cpu().numpy()
        all_preds.append(pred)
        all_probs.append(out_t.detach().cpu().numpy())
        for t in range(num_classes):
            t_ind = np.where(label_t.data.cpu().numpy() == t)
            correct_ind = np.where(pred[t_ind[0]] == t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
        size += k
    per_class_acc = per_class_correct / per_class_num

    if target_index is not None:
        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        p_prob = np.sum(all_probs[:, :num_classes - 1], 1)
        idx = np.argsort(-p_prob.flatten())[:tgt_test_data['X'].shape[0]//2]
        idx = [k for k in idx if all_preds[k] < num_classes-1]

        data = {}
        data['X'] = tgt_test_data['X'][idx]
        data['Y'] = all_preds[idx]
        with open("pred_labels/" + "Target_" + args.target + str(target_index) + ".pkl", "wb") as pkl_file:
            pickle.dump(data, pkl_file)
    return per_class_acc.mean(), per_class_acc[:-1].mean(), per_class_acc[-1]


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.isfile("data_preprocessed/{}.pkl".format(args.source)):
        raw_src_loader = load_source(args.root_dir + args.dataset + '_' + args.source + '_source_list.txt')
        save_data(raw_src_loader, name=args.source)
    src_data = pickle.load(open("data_preprocessed/{}.pkl".format(args.source), "rb"))

    Timestamps = 6
    test_data = []
    for name in ['OuterAdapter']:
        for t in range(1, Timestamps + 1):
            print("Runing on the {}-th time stamp".format(t))
            all_src_data = [src_data]

            for j in range(1, t):
                all_src_data.append(pickle.load(open("pred_labels/" + "Target_" + args.target + str(j) + ".pkl", "rb")))

            if not os.path.isfile("data_preprocessed/{}_{}.pkl".format(args.target, t)):
                raw_tgt_loader = load_target(
                    args.root_dir + args.dataset + '_' + args.target + '_' + str(t) + '_target_list.txt', timestamp=t - 1)
                raw_test_loader = load_test(
                    args.root_dir + args.dataset + '_' + args.target + '_' + str(t) + '_target_list.txt', timestamp=t - 1)
                save_data(raw_tgt_loader, name=args.target + '_{}'.format(t))
                save_data(raw_test_loader, name=args.target + '_{}'.format(t))

            tgt_data = pickle.load(open("data_preprocessed/{}_{}.pkl".format(args.target, t), "rb"))
            tgt_test_data = pickle.load(open("data_preprocessed/{}_{}.pkl".format(args.target, t), "rb"))
            test_data.append(tgt_test_data)
            train(all_src_data, tgt_data, tgt_test_data, device, target_index=t, mode_name=name)
