import warnings
warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import models
from config import opts


def main():
    # 解析参数
    args = opts.parse_opt()
    logging.info(str(args))

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # 创建分词器
    tokenizer = Tokenizer(args)

    # 创建数据加载器
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # 构建模型架构
    model_name = f"LAMRGModel_v{args.version}"
    logging.info(f"Model name: {model_name} \tModel Layers:{args.num_layers}")
    model = getattr(models, model_name)(args, tokenizer)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))

    # 获取损失函数和评估指标的函数句柄
    criterion = compute_loss
    metrics = compute_scores

    # 构建优化器和学习率调度器
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # 构建训练器并开始训练
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    logging.info(str(args))


if __name__ == '__main__':
    main()
