# Number Sence using Multimodal VAE

This repository contains the code for the framework in **Variational Mixture-of-Experts Autoencodersfor Multi-Modal Deep Generative Models** (see [paper](https://arxiv.org/pdf/1911.03393.pdf)).

## How to run

``` shell
python main.py
```

or 

``` shell
./src/script/pbs_script
```

## Requirements
List of packages we used and the version we tested the model on (see also `requirements.txt`)

```
python == 3.6.8
gensim == 3.8.1
matplotlib == 3.1.1
nltk == 3.4.5
numpy == 1.16.4
pandas == 0.25.3
scipy == 1.3.2
seaborn == 0.9.0
scikit-image == 0.15.0
torch == 1.3.1
torchnet == 0.0.4
torchvision == 0.4.2
umap-learn == 0.1.1
```

## Downloads datasets

## Pretrained network

## Usage

### Training

- **`--obj`**: Objective functions, offers 3 choices including importance-sampled ELBO (`elbo`), IWAE (`iwae`) and DReG (`dreg`, used in paper). Including the `--looser` flag when using IWAE or DReG removes unbalanced weighting of modalities, which we find to perform better empirically;
- **`--K`**: Number of particles, controls the number of particles `K` in IWAE/DReG estimator, as specified in following equation:

<p align='center'><img src="_imgs/obj.png"></p>

- **`--learn-prior`**: Prior variance learning, controls whether to enable prior variance learning. Results in our paper are produced with this enabled. Excluding this argument in the command will disable this option;
- **`--llik_scaling`**: Likelihood scaling, specifies the likelihood scaling of one of the two modalities, so that the likelihoods of two modalities contribute similarly to the lower bound. The default values are: 
    - _MNIST-SVHN_: MNIST scaling factor 32*32*3/28*28*1 = 3.92
    - _CUB Image-Cpation_: Image scaling factor 32/64*64*3 = 0.0026
- **`--latent-dimension`**: Latent dimension

You can also load from pre-trained models by specifying the path to the model folder, for example `python --model mnist_svhn --pre-trained path/to/model/folder/`. See following for the flag we used for these pretrained models:
- **MNIST-SVHN**: `--model mnist_svhn --obj dreg --K 30 --learn-prior --looser --epochs 30 --batch-size 128 --latent-dim 20`
- **CUB Image-Caption (feature)**: `--model cubISft --learn-prior --K 50 --obj dreg --looser --epochs 50 --batch-size 64 --latent-dim 64 --llik_scaling 0.002`
- **CUB Image-Caption (raw images)**: `--model cubIS --learn-prior --K 50 --obj dreg --looser --epochs 50 --batch-size 64 --latent-dim 64`

### Analysing

We offer tools to reproduce the quantitative results in our paper in `src/report`. To run any of the provided scripts, `cd` into `src`, and

- for likelihood estimation of data using a trained model, run `python calculate_likelihoods.py --save-dir path/to/trained/model/folder/ --iwae-samples 1000`;
- for coherence analysis and latent digit classification accuracy on MNIST-SVHN dataset, run `python analyse_ms.py --save-dir path/to/trained/model/folder/`;
- for coherence analysis on CUB image-caption dataset, run `python analyse_cub.py --save-dir path/to/trained/model/folder/`.
    - _**Note**_: The learnt CCA projection matrix and FastText embeddings can vary quite a bit due to the limited dataset size, therefore re-computing them as part of the analyses can result in different numeric values including for the baseline. **The relative performance of our model against the baseline remains the same, just that the numbers can different.** 
To produce similar results to what's reported in our paper, download the zip file [here](http://www.robots.ox.ac.uk/~yshi/mmdgm/pretrained_models/CCA_emb.zip) and do the following:
        1. Move `cub.all`, `cub.emb`, `cub.pc` to under `data/cub/oc:3_sl:32_s:300_w:3/`;
        2. Move the rest of the files, i.e. `emb_mean.pt`, `emb_proj.pt`, `images_mean.pt`, `im_proj.pt` to `path/to/trained/model/folder/`;
        3. Set the `RESET` variable in `src/report/analyse_cub.py` to `False`.
