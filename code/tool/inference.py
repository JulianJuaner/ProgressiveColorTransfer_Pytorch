import argparse
import os
import time
from code.configs.make_cfg import make_config
from code.configs.options import Options
from code.models.color_model import ProgressiveTransfer
from code.tool.dataset import SingleStyleDataset

from torch.utils.data import DataLoader


def inference(cfg):
    dataset = SingleStyleDataset(cfg.DATA, cfg)
    dataloader = DataLoader(
            dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
        )
    model = ProgressiveTransfer(cfg).cuda().eval()
    test_iter = iter(dataloader)
    
    for test_step in range(len(test_iter)):
        start_time = time.time()
        test_data = next(test_iter)
        res_dict = model(test_data)
        print("Time used to process the {} image".format(test_step), time.time() - start_time)
    print('end of inference.')


if "__main__" in __name__:
    # initialize exp configs.
    parser = argparse.ArgumentParser()
    OptionInit = Options(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    folder_name = opt.exp
    exp_cfg = make_config(os.path.join(folder_name, "exp.yaml"))
    print(exp_cfg)
    # inference model.
    inference(exp_cfg)
