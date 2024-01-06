# Number Sence using Multimodal VAE

This repository contains the code for "Emergence of Number Sense through the Integration of Multimodal Information." Please, see [paper](https://osf.io/preprints/psyarxiv/4bfam) for details.

The experiment code in this repository is based on "Variational Mixture-of-Experts Autoencodersfor Multi-Modal Deep Generative Models." Please, see [Shi et al. (2019)](https://arxiv.org/pdf/1911.03393.pdf).

## How to run

### Create dataset

If creating the CMNIST, OSCN, and CMNIST-OSCN dataset, execute `./data_prepare/cmnist_prepare.py` and `./data_prepare/make_index_cmnist_oscn.py`.

### Experiment settings

The experimental conditions can be set using `./src/config.py`.

### Run

Execute the following code.

``` shell
python main.py
```

### Output directory

The outputs were saved in `./rslt` directory like below.

```
.
`--- rslt
    `--- experiment_name 
        |--- synthesize
        `--- model_name 
            |--- run_id1
            |   |--- train
            |   `--- analyse
            |--- run_id2
            `--- run_id3
```

## References

Noda, K., Soda, T., & Yamashita, Y. (2024). Emergence of number sense through the integration of multimodal information: developmental learning insights from neural network models. Frontiers in Neuroscience, 18, 1330512. doi: 10.3389/fnins.2024.1330512

Shi, Y., Paige, B., & Torr, P. (2019). Variational mixture-of-experts autoencoders for multi-modal deep generative models. Advances in neural information processing systems, 32.
