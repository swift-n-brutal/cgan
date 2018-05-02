# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:54:38 2017

@author: shiwu_001
"""

import os.path as osp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser

def get_plotable_data(data, input_format='HWC'):
    data = np.clip(data, 0, 255)
    if input_format == 'CHW':
        data = data.swapaxes(0,1).swapaxes(1,2)
    data = np.require(data, dtype=np.uint8)
    return data

def draw_attr(draw, attr, pos, attr_size, attr_per_row):
    dim = attr.shape[0]
    color = {-1: 'red', 1: 'green'}
    for d in xrange(dim):
        i = d / attr_per_row
        j = d - i * attr_per_row
        x = pos[0] + j*attr_size
        y = pos[1] + i*attr_size
        draw.rectangle([x, y, x + attr_size, y + attr_size], fill=color[attr[d]], outline='black')

def display_samples(samples, save_path, input_format='NHWC',
        attr=None, attr_size=4, attr_per_row=10,
        stride=2, font_size=10, header_height=12, mid_gap=16,
        font_name='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
    real_image = samples[0]
    fake_image = samples[1]
    #
    n_samples = real_image.shape[0]
    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1)  / rows
    input_format = input_format.upper()
    idx_h = input_format.find('H')
    idx_w = input_format.find('W')
    im_h = real_image.shape[idx_h]
    im_w = real_image.shape[idx_w]
    im_h_stride = stride
    im_w_stride = stride
    # condition
    if attr is not None:
        n_attrs = attr.shape[1]
        attr_cols = attr_per_row
        attr_rows = np.ceil(n_attrs * 1. / attr_cols).astype(int)
        attr_w = attr_size * attr_cols
        assert attr_w < im_w
        attr_h = attr_size * attr_rows
        im_h_stride = attr_h + attr_size*2
    #
    half_w = (im_w + im_w_stride)*cols
    half_h = (im_h + im_h_stride)*rows
    font = ImageFont.truetype(font_name, size=font_size)
    canvas = Image.new('RGB', (half_w*2 + mid_gap, half_h + header_height), 'white')
    draw = ImageDraw.Draw(canvas)
    # header
    real_text = 'Real'
    real_header_w, real_header_h = font.getsize(real_text)
    real_header_x = (half_w - real_header_w) / 2
    real_header_y = (header_height - real_header_h) / 2
    draw.text((real_header_x, real_header_y), real_text, font=font, fill='black')
    fake_text = 'Fake'
    fake_header_w, fake_header_h = font.getsize(fake_text)
    fake_header_x = (half_w - fake_header_w) / 2 + half_w + mid_gap
    fake_header_y = (header_height - fake_header_h) / 2
    draw.text((fake_header_x, fake_header_y), fake_text, font=font, fill='black')
    for num in xrange(n_samples):
        # index of the grid to paste the images
        i = num / cols
        j = num - i*cols
        # real
        x_real = (im_w + im_w_stride)*j
        y = (im_h + im_h_stride)*i +  header_height
        im_real = Image.fromarray(real_image[num, ...])
        canvas.paste(im_real, (x_real, y))
        if attr is not None:
            draw_attr(draw, attr[num, ...], (x_real, y + im_h + attr_size), attr_size, attr_per_row)
        # fake
        x_fake = x_real + half_w + mid_gap
        im_fake = Image.fromarray(fake_image[num, ...])
        canvas.paste(im_fake, (x_fake, y))
        if attr is not None:
            draw_attr(draw, attr[num, ...], (x_fake, y + im_h + attr_size), attr_size, attr_per_row)
    canvas.save(save_path, 'PNG')

def get_parser():
    ps = ArgumentParser()
    #
    ps.add_argument('--save_path', type=str,
            default='results/sn_cgan_lr5e-05_bs64t64_ns128_fc32_maxc512_d5g1_usebn_msra/results_test_only-1.png')
    ps.add_argument('--result_path', type=str,
            default='results/sn_cgan_lr5e-05_bs64t64_ns128_fc32_maxc512_d5g1_usebn_msra/results_test_only-1.npz')
    #
    return ps

def main():
    ps = get_parser()
    args = ps.parse_args()
    fd = np.load(args.result_path)
    real_image = fd['img']
    fake_image = fd['gen_img']
    condition = fd['condition']
    display_samples([real_image, fake_image], args.save_path, attr=condition)

if __name__ == '__main__':
    main()
