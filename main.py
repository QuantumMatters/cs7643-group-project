import logging
import os
import shutil
import time
from datetime import datetime

import torch
import torchvision.utils as vutils
from sacred import Experiment
from sacred.observers import FileStorageObserver

import unir.factory.closures as closure_factory
import unir.factory.dataset as dataset_factory
import unir.factory.lr_scheduler as lr_scheduler_factory
import unir.factory.modules as module_factory
import utils.external_resources as external
from utils.meters import AverageMeter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

ex = Experiment('unsup')

def save_checkpoint(state, is_best, exp_dir, run_id, filename='checkpoint.pth.tar'):
    path = os.path.join(exp_dir, run_id, filename)
    try:
        torch.save(state, path)
        logging.info('saving to {}...'.format(path))
        if is_best:
            shutil.copyfile(path, os.path.join(exp_dir, run_id, 'model_best.pth.tar'))
    except:
        logging.warning('saving to {} FAILED'.format(path))


@ex.config
def config():
    device = "cuda"
    nepochs = 800
    time_str = datetime.now().strftime("%a-%b-%d-%H%M%S")
    exp_dir = "experiments/" + time_str
    debug = False
    use_mongo = False
    if use_mongo and not debug:
        ex.observers.append(external.get_mongo_obs())
    else:
        ex.observers.append(FileStorageObserver.create(exp_dir))
    if debug:
        nepochs = 10

    niter = {
        'train': 400,
        'test': 20,
    }
    dis_path = None
    gen_path = None
    create_datasets = dataset_factory.dataset(ex)
    create_modules = module_factory.UnsupIR(ex)
    create_closure = closure_factory.closure(ex)
    create_scheduler = lr_scheduler_factory.lr_scheduler(ex)


def write_epoch_num_file(path, epoch_num):
    """Used to record the last achieved epoch when saving models"""
    with open(path, "w") as f:
        f.write(str(epoch_num))

@ex.automain
def main(_run, nepochs, niter, device, _config, create_datasets, create_modules, create_closure, create_scheduler,
         exp_dir, gen_out_path=None, dis_out_path=None, starting_epoch=None,
        ):
    os.makedirs(exp_dir, exist_ok=True)
    shutil.make_archive('./unir', 'zip', './unir')
    ex.add_resource("./unir.zip")

    logger.info('### Dataset ###')

    dsets, corruption, nc = create_datasets()
    logger.info('### Model and Optim ###')

    mods, optims = create_modules(nc=nc, corruption=_config["corruption"], closure_name=_config['closure']['name'],
                                  gen_path=_config["gen_path"], dis_path=_config["dis_path"])
    dict_scheduler = create_scheduler(dict_optim=optims)
    closure = create_closure(mods, optims, device, measurement=corruption, scheduler=dict_scheduler)

    logger.info('### Begin Training ###')
    best_mse = float('inf')

    # create a file to record losses and write the header
    with open(f"{exp_dir}/{_run._id}/losses.csv", "w") as f:
        f.write("epoch,"
                "train_loss_G,"
                "train_loss_D,"
                "train_loss_MSE,"
                "test_loss_G,"
                "test_loss_D,"
                "test_loss_MSE\n"
            )

    start = starting_epoch or 1
    for epoch in range(start, nepochs + 1):

        # these will record the losses by minibatch and report the average across minibatches
        train_loss_Gs = AverageMeter()
        train_loss_Ds = AverageMeter()
        train_loss_MSEs = AverageMeter()
        test_loss_Gs = AverageMeter()
        test_loss_Ds = AverageMeter()
        test_loss_MSEs = AverageMeter()
        logger.info('### Starting epoch n°{} '.format(epoch))

        # there are just two iterations here: (1) train, (2) test
        for split, dl in dsets.items():

            iter = 0
            with torch.set_grad_enabled(split == 'train'):
                batch_time = AverageMeter()
                data_time = AverageMeter()
                end = time.time()

                # iterate through the minibatches
                for batch in dl:
                    print_freq = niter[split] / 5
                    for var_name, var in batch.items():
                        batch[var_name] = var.to(device)
                    data_time.update(time.time() - end)
                    closure.forward(batch)
                    if split == 'train':
                        closure.backward()
                        train_loss_Gs.update(closure.loss_G)
                        train_loss_Ds.update(closure.loss_D)
                        train_loss_MSEs.update(closure.loss_MSE)
                    else:
                        test_loss_Gs.update(closure.loss_G)
                        test_loss_Ds.update(closure.loss_D)
                        test_loss_MSEs.update(closure.loss_MSE)

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if iter % print_freq == 0:
                        logger.info('Epoch: [{0}] {split} [{1}/{2}]\t'
                                    'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Train Loss_G {train_loss_Gs.val:.3f} ({train_loss_Gs.avg:.3f})\t'
                                    'Train Loss_D {train_loss_Ds.val:.3f} ({train_loss_Ds.avg:.3f})\t'
                                    'Train Loss_MSE {train_loss_MSEs.val:.3f} ({train_loss_MSEs.avg:.3f})\t'
                                    'Test Loss_G {test_loss_Gs.val:.3f} ({test_loss_Gs.avg:.3f})\t'
                                    'Test Loss_D {test_loss_Ds.val:.3f} ({test_loss_Ds.avg:.3f})\t'
                                    'Test Loss_MSE {test_loss_MSEs.val:.3f} ({test_loss_MSEs.avg:.3f})\t'.format(
                            epoch, iter, niter[split], split=split, batch_time=batch_time, data_time=data_time,
                            train_loss_Gs=train_loss_Gs, train_loss_Ds=train_loss_Ds,
                            train_loss_MSEs=train_loss_MSEs, test_loss_Gs=test_loss_Gs, 
                            test_loss_Ds=test_loss_Ds, test_loss_MSEs=test_loss_MSEs,
                        ))
                    iter += 1

                    if iter > niter[split]:
                        break

            meters, images = closure.step()
            ims = torch.cat([v for k, v in images.items() if v.dim() == 4 and v.shape[1] in [1, 3]])
            path = f"{exp_dir}/{_run._id}/{split}_{epoch}.png"
            if epoch%50 == 0:
                with torch.no_grad():
                    vutils.save_image(ims, path, scale_each=True, normalize=True, nrow=dl.batch_size)
                logger.info("saving images in {path}".format(path=path))

            string_to_print = '*** '
            for name, v in meters.items():
                tag = 'meters' + '/' + name + '/' + split
                ex.log_scalar(tag, v, epoch)
                string_to_print += '** {name:10} {meters:.5f} '.format(split=split, name=name, meters=v)
            logger.info(string_to_print)

            if split == 'test':
                is_best = meters['loss_MSE'] < best_mse
                best_mse = max(meters['loss_MSE'], best_mse)
                _run.result = best_mse
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': mods['gen'].state_dict(),
                    'best_MSE': best_mse,
                }, is_best, _run._id, exp_dir)

        # Save models
        torch.save(mods['gen'].state_dict(), gen_out_path or f"{exp_dir}/{_run._id}/latest_gen.pth")
        torch.save(mods['dis'].state_dict(), dis_out_path or f"{exp_dir}/{_run._id}/latest_dis.pth")
        write_epoch_num_file(f"{exp_dir}/{_run._id}/epoch_num_.txt", epoch)
        print("Models saved")

        # write losses to file
        with open(f"{exp_dir}/{_run._id}/losses.csv", "a") as f:
            f.write("{epoch},"
                    "{train_loss_Gs.avg:.3f},"
                    "{train_loss_Ds.avg:.3f},"
                    "{train_loss_MSEs.avg:.3f},"
                    "{test_loss_Gs.avg:.3f},"
                    "{test_loss_Ds.avg:.3f},"
                    "{test_loss_MSEs.avg:.3f}\n".format(
                        epoch=epoch,
                        train_loss_Gs=train_loss_Gs, train_loss_Ds=train_loss_Ds,
                        train_loss_MSEs=train_loss_MSEs, test_loss_Gs=test_loss_Gs, 
                        test_loss_Ds=test_loss_Ds, test_loss_MSEs=test_loss_MSEs))
