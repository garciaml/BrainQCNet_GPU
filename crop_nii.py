#!/usr/bin/python3

import numpy as np
import nibabel as nib
import argparse
import os

parser = argparse.ArgumentParser()
# parser.add_argument('-indir', nargs=1, type=str) 
parser.add_argument('-inpath', nargs=1, type=str) 
parser.add_argument('-outdir', nargs=1, type=str) 
parser.add_argument('-filename', nargs=1, type=str) 
args = parser.parse_args()

# indir = args.indir[0]
inpath = args.inpath[0]
outdir = args.outdir[0]
filename = args.filename[0]

# proxy_img = nib.load(indir+ "/" + filename)
proxy_img = nib.load(inpath)
img = np.asarray(proxy_img.dataobj)
affine = proxy_img.affine
proxy_img.uncache()
img = np.squeeze(img)

# https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
# Mask of non-black pixels (assuming image has a single channel).
mask = img > 0

# Coordinates of non-black pixels.
coords = np.argwhere(mask)

# Bounding box of non-black pixels.
x0, y0, z0 = coords.min(axis=0)
x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top

# Get the contents of the bounding box.
cropped = img[x0:x1, y0:y1, z0:z1]

# filename = inpath.split("/")[-1]
#nib.save(filename=outdir + "/" + filename, img=nib.Nifti1Image(cropped, affine))
nib.save(filename=os.path.join(outdir, filename), img=nib.Nifti1Image(cropped, affine))

