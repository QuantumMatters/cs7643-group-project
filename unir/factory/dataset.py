import logging

from torch.utils.data import DataLoader, RandomSampler

from unir.dataset.celebA import CelebALoader
from unir.dataset.lsun import LSUNLoader
from unir.dataset.recipe import RecipeLoader
from unir.dataset.mnist import MNISTLoader
from unir.dataset.CloudSat import CloudSatLoader
from unir.module.corruption import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_default = {
    'common': {
        'batch_size': 16,
        'num_workers': 8,
        'sample': False,
        'seed': 0,
    },
    'celebA': {
        'filename': "/content/drive/MyDrive/CS7643-GroupProject/data/img_align_celeba/img_align_celeba",
        'nc': 3,
        'im_size': 64,
    },
    'LSUN': {
        'filename': '...',
        'nc': 3,
        'im_size': 64,
    },
    'recipe': {
        'filename': '...',
        'nc': 3,
        'im_size': 64,
    },
    'MNIST': {
        'filename': "...",
        'nc': 3,
        'im_size': 64,
    },
    'satellite': {
        'filename': "...",
        'nc': 3,
        'im_size': 256,
    }
}

dataset_funcs = {
    'celebA': CelebALoader,
    'LSUN': LSUNLoader,
    'recipe': RecipeLoader,
    'MNIST': MNISTLoader,
    'satellite': CloudSatLoader
}

corruption_config = {
    'common': {
    },
    'keep_patch': {
        'size_percent': (0.3, 1),
    },
    'remove_pix': {
        'percent': 0.9,
    },
    'remove_pix_dark': {
        'percent': 0.9,
    },
    'conv_noise': {
        'conv_size': 5,
        'noise_variance': 0.3,
    },
    'cloud': {
    },
}
corruption_funcs = {
    "keep_patch": KeepPatch,
    "remove_pix": RemovePixel,
    "remove_pix_dark": RemovePixelDark,
    "conv_noise": ConvNoise,
    "cloud": Cloudy
}


def corruption(ex):
    @ex.config
    def config():
        corruption = {
            'name': 'remove_pix_dark'
        }
        corruption.update(corruption_config['common'])
        corruption.update(corruption_config[corruption['name']])

    @ex.capture
    def create(corruption, im_size):
        kwargs = corruption.copy()
        name = kwargs.pop('name')
        H = corruption_funcs[name](**kwargs, im_size=im_size)
        return H

    return create


def dataset(ex):
    @ex.config
    def config():
        dataset = {
            'name': 'celebA'
            #'name': 'MNIST'
        }

        dataset.update(dataset_default['common'])
        dataset.update(dataset_default[dataset['name']])

    create_corruption = corruption(ex)

    @ex.capture
    def create(dataset):
        kwargs = dataset.copy()
        batch_size = kwargs.pop('batch_size')
        num_workers = kwargs.pop('num_workers')
        sample = kwargs.pop("sample")
        nc = kwargs.pop("nc")
        name = kwargs.pop('name')
        im_size = kwargs.pop('im_size')
        seed = kwargs.pop('seed')

        corruption = create_corruption(im_size=im_size)

        ds = dataset_funcs[name](is_train=True, measurement=corruption, **kwargs)
        test_ds = dataset_funcs[name](is_train=False, measurement=corruption, **kwargs)

        if sample:
            logger.info(f"Training set will be randomly sampled with seed {seed}")
            sampler = RandomSampler(ds, replacement=True, num_samples=batch_size)

            # for reproducibility.
            # see: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                numpy.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(seed)

            dl = DataLoader(dataset=ds,
                            batch_size=batch_size,
                            drop_last=True,
                            pin_memory=False,
                            num_workers=num_workers,
                            sampler=sampler,
                            worker_init_fn=seed_worker,
                            generator=g
                            )

        else:
            dl = DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=False,
                            num_workers=num_workers,
                            )

        test_dl = DataLoader(dataset=test_ds,
                            batch_size=dataset['batch_size'],
                            shuffle=False,
                            drop_last=True,
                            pin_memory=False,
                            num_workers=dataset['num_workers'],
                            )
        logger.info("loaded in {} : \n \t \t Train {} images (num batch {})  \n \t \t Test  {} images  (num batch {})"
                    .format(dataset["filename"], len(ds), len(dl), len(test_ds), len(test_dl)))

        return {
                   'train': dl,
                   'test': test_dl,
               }, corruption, nc

    return create
