from collections import defaultdict
import time

from PIL import Image

import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image, make_grid
import torchsummary

import src.models as models
import src.objectives as objectives
from src.utils import Logger, Timer, save_model, save_vars, unpack_data


class Runner():
    def __init__(self, args, run_path):
        self.args = args
        self.run_path = run_path

        # load model
        model_class = getattr(models, '{}'.format(args.model))
        model = model_class(args).to(args.device)
        torchsummary.summary(
            model, (model.data_size), device=args.device)

        if args.pretrained_path:
            print('Loading model {} from {}'.format(
                model.__name__, args.pretrained_path))
            model.load_state_dict(
                torch.load(args.pretrained_path + '/model.rar'))
            model._pz_params = model._pz_params
        self.model = model

        # preparation for training
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3, amsgrad=True)

        if args.model == 'smnist':
            self.train_loader, self.test_loader, self.abtest_loader = \
                self.model.getDataLoaders(
                    args.batch_size, device=args.device)
        else:
            self.train_loader, self.test_loader = \
                self.model.getDataLoaders(
                    args.batch_size, device=args.device)

        objective_name = ('m_' if hasattr(self.model, 'vaes') else '') \
            + args.obj \
            + ('_looser' if (args.looser and args.obj != 'elbo') else '')
        t_objective_name = 'cross' if 'Classifier' in self.args.model else \
            ('m_' if hasattr(self.model, 'vaes') else '') + 'iwae'
        self.objective = getattr(objectives, objective_name)
        self.t_objective = getattr(objectives, t_objective_name)
        print(
            'objectives: ', self.objective.__name__,
            '\nt_objectives: ', self.t_objective.__name__,
        )

    def train(self, epoch, agg):
        """Training models
        """
        self.model.train()
        b_loss = 0
        start_time = time.time()

        for i, dataT in enumerate(self.train_loader):
            data, label = unpack_data(
                dataT, device=self.args.device, require_label=True)

            # save_image(data[0:3], 'checks/clevr.png')
            """
            for i in range(10,20):
                # get_image(data[0][i], 'cmnist' + str(i) + '.jpg')
                tar = np.array((data[i]* 255).cpu()).T.astype(np.uint8)
                pil_image = Image.fromarray(tar)
                # pil_image.save('generated_images/test_oscn' + str(i) + '.png')
                # pil_image = Image.fromarray(
                    data[i].int().cpu().numpy().astype(np.uint8).T)
                pil_image.save('checks/oscn' + str(i) + '.png')
            """

            self.optimizer.zero_grad()
            loss = self.objective(
                self.model,
                data,
                K=self.args.K,
                conditional=self.args.use_conditional,
                labels=label,
                device=self.args.device)
            loss = -1 * loss
            """
            with open(str(self.run_path) + '/ratio.log', 'a') as f:
                print(ratio[0], ratio[1], file=f)
            """
            loss.backward()
            self.optimizer.step()
            b_loss += loss.item()
            if self.args.print_freq > 0 and i % self.args.print_freq == 0:
                print("iteration {:04d}: loss: {:6.3f}".format(
                    i, loss.item() / self.args.batch_size))

        end_time = time.time()
        agg['train_loss'].append(b_loss / len(self.train_loader.dataset))
        print('====> Epoch: {:03d} Train loss: {:.4f}'.format(
            epoch, agg['train_loss'][-1]), " took :", end_time - start_time)

    def test(self, epoch, agg):
        """Testing models
        """
        self.model.eval()
        b_loss = 0
        with torch.no_grad():
            for i, dataT in enumerate(self.test_loader):
                data, label = unpack_data(
                    dataT, device=self.args.device, require_label=True)

                self.model.reconstruct(data, self.run_path, epoch)
                # if i == 0:
                #     break
                if 'Classifier' in self.args.model:
                    loss = self.objective(
                        self.model,
                        data,
                        labels=label,
                        device=self.args.device)
                else:
                    loss = -self.t_objective(self.model, data, K=self.args.K)
                b_loss += loss.item()
                if i == 0:
                    # self.model.reconstruct(data, self.run_path, epoch)
                    print('done!')
                    break
                    # if not self.args.no_analytics:
                        # self.model.analyse(data, self.run_path, epoch)
        # agg['test_loss'].append(b_loss / len(self.test_loader.dataset))
        # print('====> Test loss: {:.4f}'.format(agg['test_loss'][-1]))

    def get_image(self, data, name):
        data = (data * 255).int()
        data = data.cpu().numpy().astype(np.uint8).T
        pil_image = Image.fromarray(data)
        pil_image.save(name)

    def estimate_log_marginal(self, K):
        """Compute an IWAE estimate of the log-marginal likelihood of test data."""
        self.model.eval()
        marginal_loglik = 0
        with torch.no_grad():
            for dataT in self.test_loader:
                data = unpack_data(dataT, device=self.args.device)
                marginal_loglik += -self.t_objective(self.model, data, K).item()

        marginal_loglik /= len(self.test_loader.dataset)
        print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))


def run_train(args, run_path):
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        runner = Runner(args=args, run_path=run_path)

        for epoch in range(1, args.epochs + 1):
            runner.train(epoch, agg)
            save_model(runner.model, run_path + '/model.rar')
            save_vars(agg, run_path + '/losses.rar')
            runner.model.generate(run_path, epoch)
            runner.test(epoch, agg)

        if args.logp:
            # compute as tight a marginal likelihood as possible
            runner.estimate_log_marginal(5000)

if __name__ == '__main__':
    args = {
        'run_type': 'train',
    }
    run_path = './rslt/test'
    run_train(args=args, run_path=run_path)
