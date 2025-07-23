import warnings
warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tester import Tester
from modules.loss import compute_loss
import models
from config import opts


def main():

    args = opts.parse_opt()
    logging.info(str(args))


    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)


    tokenizer = Tokenizer(args)

    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)


    model_name = f"LAMRGModel_v{args.version}"
    logging.info(f"Model name: {model_name} \tModel Layers:{args.num_layers}")
    model = getattr(models, model_name)(args, tokenizer)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))


    criterion = compute_loss
    metrics = compute_scores


    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    tester = Tester(model, criterion, metrics, optimizer, args, lr_scheduler, test_dataloader)
    tester.test_step()
    logging.info(str(args))


if __name__ == '__main__':
    main()
