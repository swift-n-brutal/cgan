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

- Download [tfbox](https://github.com/swift-n-brutal/tfbox) and append its path to 'PYTHONPATH'
'''
git clone https://github.com/swift-n-brutal/tfbox
export PYTHONPATH=$PYTHONPATH:/path/to/tfbox
'''

## Training CGAN on CelebA Dataset
- Download (CelebA)[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html] dataset with attributes annotations.
- Training command
'''
CUDA_VISIBLE_DEVICES=0 python solver_sncgan.py --folder /path/to/img_align_celeba --names /path/to/list_attr_celeba.txt
'''
