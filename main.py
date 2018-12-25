from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##################
# Pytorch imports
##################
import torch.nn.functional as F

#################
# Python imports
#################
import os
import time
import argparse
import threading
import numpy as np
from tqdm import tqdm as tq

###############
# Tune imports
###############
import ray
from hyperopt import hp
from ray.tune.suggest import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.util import pin_in_object_store, get_pinned_object
from ray.tune import Trainable, run_experiments, register_trainable, Experiment

################
# local imports
################
from utils import *


pinned_obj_dict = {}


class TrainerClass(Trainable):
    def _setup(self, config):
        self.data_loader_train = get_pinned_object(pinned_obj_dict['data_loader_train'])
        self.data_loader_valid = get_pinned_object(pinned_obj_dict['data_loader_valid'])
        self.cuda_available = torch.cuda.is_available()
        print("Cuda is available: {}".format(self.cuda_available))
        self.model = get_model()
        if self.cuda_available:
            self.model.cuda()
        opt = getattr(torch.optim, self.config['optimizer'])
        self.optimizer = opt(self.model.classifier[6].parameters(), lr=self.config['lr'])
        self.batch_accumulation = self.config['batch_accumulation']

    def _train_iter(self):
        j = 1
        i = 0
        self.model.train()
        self.optimizer.zero_grad()
        progress_bar = tq(self.data_loader_train)
        progress_bar.set_description("Training")
        avg_loss = 0.0
        for batch_idx, (data, target) in enumerate(progress_bar, 1):
            if self.cuda_available:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if j % self.batch_accumulation == 0:
                j = 1
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                j += 1
            if i == (len(self.data_loader_train) / 2): break
            i += 1
            avg_loss += loss.item()
            if batch_idx % (len(self.data_loader_train) / 8) == 0:
                progress_bar.set_postfix({'Loss': '{:.3f}'.format((avg_loss / (batch_idx * target.shape[0])))})
        torch.cuda.empty_cache()

    def _valid(self):
        self.model.eval()
        avg_loss = 0.0
        avg_acc = 0.0
        progress_bar = tq(self.data_loader_valid)
        progress_bar.set_description("Validation")
        for batch_idx, (data, target) in enumerate(progress_bar, 1):
            if self.cuda_available:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            avg_loss += loss
            y_hat = output.argmax(dim=1)
            avg_acc += (target == y_hat).sum().item()
            if batch_idx % (len(self.data_loader_valid) / 8) == 0:
                l = avg_loss / (batch_idx * target.shape[0])
                acc = avg_acc / (batch_idx * target.shape[0])
                metrics = {
                    'loss': '{:.3f}'.format(l.item()),
                    'acc': '{:.2f}%'.format(acc)
                }
                progress_bar.set_postfix(metrics)
        loss = avg_loss / len(self.data_loader_valid.dataset)
        acc = avg_acc / len(self.data_loader_valid.dataset)
        torch.cuda.empty_cache()
        return {"loss": loss.detach().cpu().item(), "acc": acc}

    def _train(self):
        self._train_iter()
        return self._valid()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    ray.init()

    t_loader, v_loader = get_loaders(train_batch_size=16)
    pinned_obj_dict['data_loader_train'] = pin_in_object_store(t_loader)
    pinned_obj_dict['data_loader_valid'] = pin_in_object_store(v_loader)

    trainable_name = 'hyp_search_train'
    register_trainable(trainable_name, TrainerClass)

    reward_attr = "acc"

    #############################
    # Define hyperband scheduler
    #############################
    hpb = AsyncHyperBandScheduler(time_attr="training_iteration",
                                  reward_attr=reward_attr,
                                  grace_period=40,
                                  max_t=300)

    ##############################
    # Define hyperopt search algo
    ##############################
    space = {
        'lr': hp.uniform('lr', 0.001, 0.1),
        'optimizer': hp.choice("optimizer", ['Adam', 'SGD', 'Adadelta']),
        'batch_accumulation': hp.choice("batch_accumulation", [4, 8, 16])
    }
    hos = HyperOptSearch(space, max_concurrent=4, reward_attr=reward_attr)

    #####################
    # Define experiments
    #####################
    exp_name = "hyp_search_hyperband_hyperopt_{}".format(time.strftime("%Y-%m-%d_%H.%M.%S"))
    exp = Experiment(
        name=exp_name,
        run=trainable_name,
        num_samples=10,  # the number of experiments
        trial_resources={
            "cpu": args.numCpu,
            "gpu": args.numGpu
        },
        checkpoint_freq=3,
        checkpoint_at_end=True,
        stop={
            reward_attr: 0.95,
            "training_iteration": 9,  # how many times a specific config will be trained
        }
    )

    ##################
    # Run tensorboard
    ##################
    if args.runTensorBoard:
        thread = threading.Thread(target=launch_tensorboard, args=[exp_name])
        thread.start()
        launch_tensorboard(exp_name)

    ##################
    # Run experiments
    ##################
    run_experiments(exp, search_alg=hos, scheduler=hpb, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--seed', type=int, default=41, help='Set the randome generators seed (default: 41)')
    parser.add_argument('-c', '--numCpu', type=int, default=4, help='Set the number of CPU to use (default: 4)')
    parser.add_argument('-g', '--numGpu', type=int, default=0, help='Set the number of GPU to use (default: 0)')
    parser.add_argument('-t', '--runTensorBoard', action='store_true', help='Run tensorboard (default: false)')
    args = parser.parse_args()
    main(args)
