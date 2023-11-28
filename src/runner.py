from collections import defaultdict
import time

from PIL import Image

import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image, make_grid
import torchsummary

import src.models
import src.objectives
from src.utils import Logger, Timer, save_model, save_vars, unpack_data

class Runner():
    def __init__(args):
        # load model
        model_class = getattr(models, '{}'.format(args.model))
        model = model_class(args).to(args.device)
        torchsummary.summary(model)
        if pretrained_path:
            # print('Loading model {} from {}'.format(model.modelName, pretrained_path))
            print('Loading model {} from {}'.format(args.model, pretrained_path))
            model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
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

        self.objective = getattr(
            objectives,
            ('m_' if hasattr(self.model, 'vaes') else '')
            + args.obj
            + ('_looser' if (args.looser and args.obj != 'elbo') else ''))
        self.t_objective = getattr(
            objectives,
            ('m_' if hasattr(self.model, 'vaes') else '') + 'iwae')


    def train(epoch, agg):
        """Training models
        """
        self.model.train()
        b_loss = 0
        start_time = time.time()

        for i, dataT in enumerate(train_loader):
            data, label = unpack_data(
                dataT, device=args.device, require_label=True)

            #save_image(data[0:3], 'checks/clevr.png')
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
                K=args.K,
                conditional=args.use_conditional,
                labels=label,
                device=args.device)
            loss = -1 * loss
            """
            with open(str(runPath) + '/ratio.log', 'a') as f:
                print(ratio[0], ratio[1], file=f)
            """
            loss.backward()
            self.optimizer.step()
            b_loss += loss.item()
            if args.print_freq > 0 and i % args.print_freq == 0:
                print("iteration {:04d}: loss: {:6.3f}".format(
                    i, loss.item() / args.batch_size))

        end_time = time.time()
        agg['train_loss'].append(b_loss / len(train_loader.dataset))
        print('====> Epoch: {:03d} Train loss: {:.4f}'.format(
            epoch, agg['train_loss'][-1]), " took :", end_time - start_time)

    def test(epoch, agg):
        """Testing models
        """
        model.eval()
        b_loss = 0
        with torch.no_grad():
            for i, dataT in enumerate(test_loader):
                data, label = unpack_data(
                    dataT, device=args.device, require_label=True)
                model.reconstruct(data, runPath, epoch)
                if i == 0:
                    break

                loss = -t_objective(model, data, K=args.K)
                b_loss += loss.item()
                if i == 0:
                    # model.reconstruct(data, runPath, epoch)
                    print('done!')
                    break
                    # if not args.no_analytics:
                        # model.analyse(data, runPath, epoch)
        # agg['test_loss'].append(b_loss / len(test_loader.dataset))
        # print('====> Test loss: {:.4f}'.format(agg['test_loss'][-1]))

    def get_image(data, name):
        data = (data * 255).int()
        data = data.cpu().numpy().astype(np.uint8).T
        pil_image = Image.fromarray(data)
        pil_image.save(name)

    def estimate_log_marginal(K):
        """Compute an IWAE estimate of the log-marginal likelihood of test data."""
        model.eval()
        marginal_loglik = 0
        with torch.no_grad():
            for dataT in test_loader:
                data = unpack_data(dataT, device=args.device)
                marginal_loglik += -t_objective(model, data, K).item()

        marginal_loglik /= len(test_loader.dataset)
        print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))


def main(args, runPath):
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        runner = Runner(args=args)

        for epoch in range(1, args.epochs + 1):
            runner.train(epoch, agg)
            save_model(runner.model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')
            runner.model.generate(runPath, epoch)
            runner.test(epoch, agg)

        if args.logp:
            # compute as tight a marginal likelihood as possible
            runner.estimate_log_marginal(5000)

if __name__ == '__main__':
    args = {
        'run_type': 'train',
    }
    runPath = './rslt/test'
    main(args=args, runPath=runPath)