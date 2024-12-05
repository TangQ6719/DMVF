import os
import logging
from abc import abstractmethod
import json
import numpy as np
import time
import torch
import pandas as pd
from scipy import sparse
from numpy import inf
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .loss import compute_loss_g

METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDEr', 'METEOR','ROUGE_L']


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        # 初始化训练器的参数
        self.args = args
        # tensorboard 记录参数和结果
        self.writer = SummaryWriter(args.save_dir)
        self.print_args2tensorbord()
        self.logger = logging.getLogger(__name__)
        # 配置GPU设备，将模型移动到指定的设备上
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.criterion_g = compute_loss_g
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self._load_checkpoint(args.load)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch_g(self, epoch):
        raise NotImplementedError

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                logging.info(f'==>> Model lr: {self.optimizer.param_groups[1]["lr"]:.7}, '
                             f'Visual Encoder lr: {self.optimizer.param_groups[0]["lr"]:.7}')
                result = self._train_epoch(epoch)
                # if self.args.version == '12' :
                #     print(111)
                #     result = self._train_epoch(epoch)
                # else:
                #     print(222)
                #     result = self._train_epoch_g(epoch)

                # 将训练结果记录到日志中
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)

                # 打印训练结果到屏幕
                self._print_epoch(log)

                # 根据配置的指标评估模型性能，保存最佳的模型检查点作为model_best
                improved = False
                if self.mnt_mode != 'off':
                    try:
                        # 根据配置的指标评估模型性能，保存最佳的模型检查点作为model_best
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        logging.error(
                            "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                                self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0

                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        logging.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=improved)
            except KeyboardInterrupt:
                logging.info('=> User Stop!')
                self._save_checkpoint(epoch, save_best=False, interrupt=True)
                logging.info('Saved checkpint!')
                if epoch > 1:
                    self._print_best()
                    self._print_best_to_file()
                return

        self._print_best()
        self._print_best_to_file()

    def print_args2tensorbord(self):
        for k, v in vars(self.args).items():
            self.writer.add_text(k, str(v))

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        for split in ['val', 'test']:
            self.best_recorder[split]['version'] = f'V{self.args.version}'
            self.best_recorder[split]['visual_extractor'] = self.args.visual_extractor
            self.best_recorder[split]['time'] = crt_time
            self.best_recorder[split]['seed'] = self.args.seed
            self.best_recorder[split]['best_model_from'] = 'val'
            self.best_recorder[split]['lr'] = self.args.lr_ed

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            logging.info("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            logging.info(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, interrupt=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if interrupt:
            filename = os.path.join(self.checkpoint_dir, 'interrupt_checkpoint.pth')
        else:
            filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        logging.debug("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            logging.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        logging.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        logging.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)
            self.writer.add_text(f'best_BELU4_byVal', str(log["test_BLEU_4"]), log["epoch"])

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)
            # self.writer.add_text(f'best_val_BELU4', str(log["val_BLEU_4"]), log["epoch"])
            self.writer.add_text(f'best_BELU4_byTest', str(log["test_BLEU_4"]), log["epoch"])

    def _print_best(self):
        logging.info('\n' + '*' * 20 + 'Best results' + '*' * 20)
        logging.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        self._prin_metrics(self.best_recorder['val'], summary=True)

        logging.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        self._prin_metrics(self.best_recorder['test'], summary=True)

        # For Record
        print(self.checkpoint_dir)
        vlog, tlog = self.best_recorder['val'], self.best_recorder['test']
        if 'epoch' in vlog:
            print(f'Val  set: Epoch: {vlog["epoch"]} | ' + 'loss: {:.4} | '.format(vlog["train_loss"]) + ' | '.join(
                ['{}: {:.4}'.format(m, vlog['test_' + m]) for m in METRICS]))
            print(f'Test Set: Epoch: {tlog["epoch"]} | ' + 'loss: {:.4} | '.format(tlog["train_loss"]) + ' | '.join(
                ['{}: {:.4}'.format(m, tlog['test_' + m]) for m in METRICS]))
            print(','.join(['{:.4}'.format(vlog['test_' + m]) for m in METRICS]) + f',E={vlog["epoch"]}'
                  + f'|TE={tlog["epoch"]} B4={tlog["test_BLEU_4"]:.4}')

    def _prin_metrics(self, log, summary=False):
        if 'epoch' not in log:
            logging.info("===>> There are not Best Results during this time running!")
            return
        logging.info(
            f'VAL ||| Epoch: {log["epoch"]}|||' + 'train_loss: {:.4}||| '.format(log["train_loss"]) + ' |||'.join(
                ['{}: {:.4}'.format(m, log['val_' + m]) for m in METRICS]))
        logging.info(
            f'TEST || Epoch: {log["epoch"]}|||' + 'train_loss: {:.4}||| '.format(log["train_loss"]) + ' |||'.join(
                ['{}: {:.4}'.format(m, log['test_' + m]) for m in METRICS]))

        if not summary:
            if isinstance(log['epoch'], str):
                epoch_split = log['epoch'].split('-')
                e = int(epoch_split[0])
                if len(epoch_split) > 1:
                    it = int(epoch_split[1])
                    epoch = len(self.train_dataloader) * e + it
                else:
                    epoch = len(self.train_dataloader) * e
            else:
                epoch = int(log['epoch']) * len(self.train_dataloader)

            for m in METRICS:
                self.writer.add_scalar(f'val/{m}', log["val_" + m], epoch)
                self.writer.add_scalar(f'test/{m}', log["test_" + m], epoch)

    def _output_generation(self, predictions, gts, idxs, epoch, iters=0, split='val'):
        # from nltk.translate.bleu_score import sentence_bleu
        output = list()
        for idx, pre, gt in zip(idxs, predictions, gts):
            # score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt})

        # output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        json_file = f'Enc2Dec-{epoch}_{iters}_{split}_generated.json'
        output_filename = os.path.join(self.checkpoint_dir, json_file)
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False)

    def _output_generation_g(self, predictions, gts, idxs, epoch):
        # from nltk.translate.bleu_score import sentence_bleu
        output = list()
        for idx, pre, gt in zip(idxs, predictions, gts):
            # score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt})

        # output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        json_file = f'Enc2Dec-{epoch}_generated.json'
        output_filename = os.path.join(self.checkpoint_dir, json_file)
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False)

    def _print_epoch(self, log):
        logging.info(f"Epoch [{log['epoch']}/{self.epochs}] - {self.checkpoint_dir}")
        self._prin_metrics(log)


class Tester(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.test_dataloader = test_dataloader

    def test_step(self, mode='test'):
        # 在验证集或测试集上评估模型
        ilog = {}
        log = dict()
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res, val_idxs = [], [], []
            t = tqdm(self.test_dataloader, ncols=80)
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
                images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                                                             reports_masks.to(self.device), labels.to(self.device)
                outputs = self.model(images_id, images, mode='sample')
                # 解码模型输出得到生成的报告
                reports = self.model.tokenizer.decode_batch(outputs[0].cpu().numpy())
                # 解码真实报告
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                # 将生成的报告、真实报告和图像ID添加到对应的列表中
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                val_idxs.extend(images_id)
            # 使用评估指标计算模型性能
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            ilog.update(**{f'{mode}_' + k: v for k, v in val_met.items()})
            # 输出生成的报告、真实报告和图像ID
            # self._output_generation(val_res, val_gts, val_idxs, epoch, iters, mode)
            log.update(**{'test_' + k: v for k, v in val_met.items()})
            print(log)
        return ilog

