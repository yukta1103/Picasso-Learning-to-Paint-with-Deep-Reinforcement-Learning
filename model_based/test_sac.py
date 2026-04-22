import os
import cv2
import torch
import numpy as np
import argparse

from DRL.actor_sac import ActorSAC
from Renderer.stroke_gen import *
from Renderer.model import FCN

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser()
parser.add_argument('--max_step',  default=40,                                    type=int)
parser.add_argument('--actor',     default='./model/Paint-sac-run1/actor_sac.pkl',type=str)
parser.add_argument('--renderer',  default='./renderer.pkl',                      type=str)
parser.add_argument('--img',       default='image/test.png',                      type=str)
parser.add_argument('--imgid',     default=0,                                     type=int)
parser.add_argument('--divide',    default=4,                                     type=int)
args = parser.parse_args()

canvas_cnt    = args.divide * args.divide
T             = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

img = cv2.imread(args.img, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f'Cannot read image: {args.img}')
origin_shape = (img.shape[1], img.shape[0])

# CoordConv grid
coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device)

# Load renderer
Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer, map_location=device))
Decoder = Decoder.to(device).eval()

def decode(x, canvas):
    x = x.view(-1, 13)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke       = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke       = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

def small2large(x):
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    return x.reshape(args.divide * width, args.divide * width, -1)

def large2small(x):
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    return x.reshape(canvas_cnt, width, width, 3)

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx in (0, args.divide*width-1) or ty in (0, args.divide*width-1):
            return img
        img[tx,ty] = (img[tx,ty] + img[tx+1,ty] + img[tx,ty+1] +
                      img[tx-1,ty] + img[tx,ty-1] + img[tx+1,ty-1] +
                      img[tx-1,ty+1] + img[tx-1,ty-1] + img[tx+1,ty+1]) / 9
        return img
    for p in range(args.divide):
        for q in range(args.divide):
            x, y = p * width, q * width
            for k in range(width):
                img = smooth_pix(img, x+k, y+width-1)
                if q != args.divide-1:
                    img = smooth_pix(img, x+k, y+width)
            for k in range(width):
                img = smooth_pix(img, x+width-1, y+k)
                if p != args.divide-1:
                    img = smooth_pix(img, x+width, y+k)
    return img

def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    cv2.imwrite('output_sac/generated{}.png'.format(imgid), output)

# Load SAC actor
actor = ActorSAC(num_inputs=9, depth=18, action_dim=65)
actor.load_state_dict(torch.load(args.actor, map_location=device))
actor = actor.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)

patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

img_t = cv2.resize(img, (width, width))
img_t = img_t.reshape(1, width, width, 3)
img_t = np.transpose(img_t, (0, 3, 1, 2))
img_t = torch.tensor(img_t).to(device).float() / 255.

os.makedirs('output_sac', exist_ok=True)

with torch.no_grad():
    if args.divide != 1:
        args.max_step = args.max_step // 2

    for i in range(args.max_step):
        stepnum  = T * i / args.max_step          # (1,1,128,128)
        actor_in = torch.cat([canvas, img_t, stepnum, coord], dim=1)  # (1,9,128,128)
        actions  = actor.act(actor_in)             # deterministic mean
        canvas, res = decode(actions, canvas)
        print('step {}, L2={:.5f}'.format(i, ((canvas - img_t) ** 2).mean()))
        for j in range(5):
            save_img(res[j], args.imgid)
            args.imgid += 1

    if args.divide != 1:
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        coord_  = coord.expand(canvas_cnt, 2, width, width)
        T_      = T.expand(canvas_cnt, 1, width, width)

        for i in range(args.max_step):
            stepnum  = T_ * i / args.max_step
            actor_in = torch.cat([canvas, patch_img, stepnum, coord_], dim=1)
            actions  = actor.act(actor_in)
            canvas, res = decode(actions, canvas)
            print('divided step {}, L2={:.5f}'.format(i, ((canvas - patch_img) ** 2).mean()))
            for j in range(5):
                save_img(res[j], args.imgid, divide=True)
                args.imgid += 1