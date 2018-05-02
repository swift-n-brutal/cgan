[//]: <links>
[sngans]: https://openreview.net/forum?id=B1QRgziT-
[pcgans]: https://openreview.net/forum?id=ByS1VpgRZ
[celeba]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# CGAN
Implementation of Conditional Generative Adversarial Networks using [tfbox](https://github.com/swift-n-brutal/tfbox).

## References
- Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. *Spectral Normalization for Generative Adversarial Networks*. ICLR2018. [OpenReview][sngans]
- Takeru Miyato, Masanori Koyama. *cGANs with Projection Discriminator*. ICLR2018. [OpenReview][pcgans]
- Ziwei Liu and Ping Luo and Xiaogang Wang and Xiaoou Tang. *Deep Learning Face Attributes in the Wild*. ICCV2015. [Project][celeba]

## Setup

- Download [tfbox](https://github.com/swift-n-brutal/tfbox) and append its path to `PYTHONPATH`
```
git clone https://github.com/swift-n-brutal/tfbox
export PYTHONPATH=$PYTHONPATH:/path/to/tfbox
```

## Training CGAN on CelebA Dataset
- Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset with attributes annotations.
- Training command
```
CUDA_VISIBLE_DEVICES=0 python solver_sncgan.py --folder /path/to/img_align_celeba --names /path/to/list_attr_celeba.txt
```

### Results
- We trained the networks using default parameter, except for `--lr 0.00005`, for around 12 hours, and obtained the following results
- The attributes annotation is shown right below each image, with green for 1 and red for -1, in order as they are listed in 'list_attr_celeba.txt'.

<img src="https://github.com/swift-n-brutal/cgan/blob/master/results/sn_cgan_lr5e-05_bs64t64_ns128_fc32_maxc512_d5g1_usebn_msra/results_test_only-1.png" width="1072">
