import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
# from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import utils
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib2 import Path
# from wiki_loader import WikipediaDataSet
from wiki_dataset import WikipediaDataSet, collate_fn
import accuracy
import numpy as np
from termcolor import colored
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import os
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# 设置随机数种子
setup_seed(42)

torch.multiprocessing.set_sharing_strategy('file_system')
writer = SummaryWriter()
preds_stats = utils.predictions_analysis()
fake_sent = "fake sent 123: bla one bla day bla whatever."


def import_model(model_name, args):
    module = __import__('models.' + model_name, fromlist=['models'])
    # return module.create()
    return module.get_model(args)


class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold


def train(model, args, epoch, dataset, logger, optimizer, dev_dl):
    model.train()
    total_loss = float(0)
    # writer = SummaryWriter()
    # writer.add_graph(model)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break

                pbar.update()
                # model.zero_grad()
                # print('========train input:', len(data), len(data[0]), len(data[0][0]), len(target[0]))
                target_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                if args.AUX:
                    output, loss2 = model(data)
                    # print(len(target), target)
                    blk_len = []
                    for s_blk in data:
                        if fake_sent in s_blk:
                            idx = s_blk.index(fake_sent) - 1
                        else:
                            idx = len(s_blk)-1

                        blk_len.append(idx)
                    # print(len(blk_len), blk_len)
                    aux_label = torch.LongTensor([torch.sum(target[i][:blk_len[i]]) for i in range(len(target))])
                    # print(aux_label.shape, aux_label)
                    a = 0.1
                    loss = model.criterion(output, target_var) + model.criterion(loss2, aux_label.cuda()) * a
                else:
                    output = model(data)
                    loss = model.criterion(output, target_var)
                # print('train target:', output.shape, target_var.shape)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                '''
                if i % 1000 == 0 and i != 0:
                    val_pk, threshold = validate(model, args, epoch, dev_dl, logger)
                    # break
                '''
                total_loss += loss.item()

                writer.add_scalar('Training Loss', loss.item(), i+epoch*len(dataset))
                pbar.set_description('Training, loss={:.4}'.format(loss.item()))
    # writer.close()
    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    # log_value('Training Loss', total_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target) in enumerate(dataset):
            with torch.no_grad():
                if i == args.stop_after:
                    break
                pbar.update()

                win_num = len(data[0])

                mini_bs = 128
                '''
                for n in range(0, win_num, mini_bs):
                    inputs.append(data[0][n:n+mini_bs])
                '''
                # print('inputs:', len(inputs), len(inputs[0]), len(data[0]), win_num)
                output_temp = []
                for k in range(0, win_num, mini_bs):
                    output = model(data[0][k:k+mini_bs])
                    output_temp.extend(output)
                output = torch.stack(output_temp, 0)

                # print(output.shape)
                targets_var = torch.cat(target, 0)
                target_seg = targets_var.numpy()
                output_prob = F.softmax(output, 1)
                output_preds = output_prob.data.cpu().numpy()
                # print('output_softmax, target:', output_preds.shape, target_seg.shape)

                scores = [[] for _ in range(len(target_seg)+1)]
                for blk_id in range(win_num):
                    for sent_id in range(args.window):
                        # idx = blk_id*16 + sent_id
                        idx = blk_id + sent_id
                        if data[0][blk_id][sent_id] == fake_sent:
                            assert idx-1 == len(target_seg)
                            break
                        scores[idx].append(output_preds[blk_id*args.window+sent_id])
                # print('scores:', len(scores), end_idx, len(scores[0]))
                output_p = np.array([np.mean(x, axis=0) for x in scores[:-1]])
                # print(len(output_p), len(output_p[0]), output_p[1])
                
                output_seg = output_p.argmax(axis=1)
                # print('seg:', output_seg.shape, target_seg.shape, output_p.shape, len(target[0]))
                preds_stats.add(output_seg, target_seg)

                acc.update(output_p, target)

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk, threshold


def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = accuracy.Accuracy()
        for i, (data, target) in enumerate(dataset):
            with torch.no_grad():
                if i == args.stop_after:
                    break
                pbar.update()

                win_num = len(data[0])
                mini_bs = 128
                output_temp = []
                for k in range(0, win_num, mini_bs):
                    output = model(data[0][k:k + mini_bs])
                    output_temp.extend(output)
                output = torch.stack(output_temp, 0)

                output_prob = F.softmax(output, 1)
                output_preds = output_prob.data.cpu().numpy()
                targets_var = torch.cat(target, 0)
                target_seg = targets_var.numpy()

                scores = [[] for _ in range(len(target_seg)+1)]

                for blk_id in range(win_num):
                    for sent_id in range(args.window):
                        # idx = blk_id*16 + sent_id
                        idx = blk_id + sent_id
                        if data[0][blk_id][sent_id] == fake_sent:
                            assert idx-1 == len(target_seg)
                            break

                        scores[idx].append(output_preds[blk_id * args.window + sent_id])
                output_p = np.array([np.mean(x, axis=0) for x in scores[:-1]])

                output_seg = output_p[:, 1] > threshold
                output_seg = np.array(output_seg, dtype=int)

                preds_stats.add(output_seg, target_seg)

                h = np.append(output_seg, [1])
                tt = np.append(target_seg, [1])

                acc.update(h, tt)

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.info('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk,
                                                                                                          epoch_windiff,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk


# @record
def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    # utils.read_config_file(args.config)
    # utils.config.update(args.__dict__)
    # logger.debug('Running with config %s', utils.config)

    # configure(os.path.join('runs', args.expname))
    '''
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda', args.local_rank)
    '''
    if not args.infer:
        if args.wiki:
            # dataset_path = Path(utils.config['wikidataset'])
            train_dataset = WikipediaDataSet('data/wiki_727/train16-8.pkl', train=True, high_granularity=args.high_granularity)
            dev_dataset = WikipediaDataSet('data/wiki_727/dev16-1.pkl', train=False, high_granularity=args.high_granularity)
            test_dataset = WikipediaDataSet('data/wiki_727/test16-1.pkl', train=False, high_granularity=args.high_granularity)

        else:
            dataset_path = utils.config['choidataset']
            train_dataset = ChoiDataset(dataset_path, word2vec)
            dev_dataset = ChoiDataset(dataset_path, word2vec)
            test_dataset = ChoiDataset(dataset_path, word2vec)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True, num_workers=args.num_workers)
    dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False, num_workers=args.num_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False, num_workers=args.num_workers)

    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set

    if args.model:
        model = import_model(args.model, args)
    elif args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)

    model = maybe_cuda(model, args.cuda)
    # model = model.to(local_rank)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if not args.infer:
        best_val_pk = 1.0
        for j in range(args.epochs):

            train(model, args, j, train_dl, logger, optimizer, dev_dl)

            with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
                torch.save(model, f)

            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            if val_pk < best_val_pk and val_pk < 0.20:
                test_pk = test(model, args, j, test_dl, logger, threshold)
                logger.debug(
                    colored(
                        'Current best model from epoch {} with p_k {} and threshold {}'.format(j, test_pk, threshold),
                        'green'))
                best_val_pk = val_pk
                with (checkpoint_path / 'best_model.t7'.format(j)).open('wb') as f:
                    torch.save(model, f)

    else:
        test_dataset = WikipediaDataSet(args.infer, high_granularity=args.high_granularity)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        print(test(model, args, 0, test_dl, logger, 0.4))
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true', default=True)
    parser.add_argument('--train', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=32)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=1)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    # parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='inference_dir', type=str)
    # parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--AUX', help='multi tasks', action='store_true')

    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_pos', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=3072)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--max_len_seq', type=int, default=32)

    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)

    opt = parser.parse_args()
    '''
    opt.device = 0 if opt.cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    '''
    main(opt)
