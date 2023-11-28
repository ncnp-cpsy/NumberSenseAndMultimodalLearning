from .mmvae_cmnist_oscn import CMNIST_OSCN as MMVAE_CMNIST_OSCN
from .vae_mnist import MNIST as VAE_MNIST
from .vae_cmnist import CMNIST as VAE_CMNIST
from .vae_oscn import OSCN as VAE_OSCN
from .classifier_oscn import ClassifierOSCN as Classifier_OSCN
from .classifier_cmnist import ClassifierCMNIST as Classifier_CMNIST

# from .mmvae_cub_images_sentences import CUB_Image_Sentence as VAE_cubIS
# from .mmvae_cub_images_sentences_ft import CUB_Image_Sentence_ft as VAE_cubISft
# from .mmvae_mnist_svhn import MNIST_SVHN as VAE_mnist_svhn
# from .mmvae_mnist_clevr import MNIST_CLEVR as VAE_mnist_clevr
# from .vae_cub_image import CUB_Image as VAE_cubI
# from .vae_cub_image_ft import CUB_Image_ft as VAE_cubIft
# from .vae_cub_sent import CUB_Sentence as VAE_cubS
# from .vae_svhn import SVHN as VAE_svhn
# from .vae_smnist import SMNIST as VAE_smnist
# from .vae_clevr import CLEVR as VAE_clevr

__all__ = [
    VAE_MNIST,
    VAE_CMNIST,
    VAE_OSCN,
    MMVAE_CMNIST_OSCN,
    Classifier_CMNIST,
    Classifier_OSCN,
]
