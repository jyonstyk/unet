import os
import glob
import cv2
import numpy as np
from itertools import chain
import random

print("enter [image directory(.tif)] and [label directory(.tif)]")

x_name,y_name = input().split()

# file searching
x_image_names_from = "../data/**/{}/**/*.tif".format(x_name)
y_image_names_from = "../data/**/{}/**/*.tif".format(y_name)

x_image_names = glob.glob(x_image_names_from)
y_image_names = glob.glob(y_image_names_from)

print("select {} images".format(len(x_image_names)))

# file reading 
x = np.array([cv2.imread(x,0)for x in np.sort(x_image_names)])
y = np.array([cv2.imread(y,0)for y in np.sort(y_image_names)])

# binning x and y いる？
bin_ratio = 2
x = np.array([cv2.resize(x_slice,(x.shape[2]//bin_ratio,x.shape[1]//bin_ratio)) for x_slice in x]) # binning x
y = np.array([cv2.resize(y_slice,(y.shape[2]//bin_ratio,y.shape[1]//bin_ratio)) for y_slice in y]) # binning y

# 各種変数の定義
one_page_size = 256
input_pages = x.shape[0] # 画像枚数
image_height = y.shape[1]
image_width = y.shape[2]

# ラベルのある部分の周り128マスだけ取ってくる処理

# 輪郭抽出
contours = [cv2.findContours(y_slice,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0] for y_slice in y] # 各画像ごとのラベルの輪郭
contours_count = [np.shape(contour)[0] for contour in contours] # 一枚の中の物体数

# 輪郭の平坦化(１つの画像に複数の輪郭が含まれている場合別々にカウント)
contours_flatten_page_index = list(chain.from_iterable([np.ones(c,dtype="int8")*i for i,c in enumerate(contours_count)]))
contours_flatten_label_index = list(chain.from_iterable([np.arange(0,c) for c in contours_count]))
rects_flatten = [cv2.boundingRect(contours[p][l]) for p,l in zip(contours_flatten_page_index,contours_flatten_label_index)]

# 大きすぎる輪郭を除外
suitable_size_index = [n for n,rect in enumerate(rects_flatten) if rect[2] < one_page_size and rect[3] < one_page_size]
contours_flatten_page_index_filt = [contours_flatten_page_index[s] for s in suitable_size_index]
contours_flatten_label_index_filt = [contours_flatten_label_index[s] for s in suitable_size_index]
rects_flatten_filt = [rects_flatten[s] for s in suitable_size_index]

# 各輪郭の罫線からの距離を計算

dist_from_border = [
    (min(x,one_page_size-w),min(image_width-x-w,one_page_size-w),min(y,one_page_size-h),min(image_height-y-h,one_page_size-h),x,y) for x,y,w,h in rects_flatten_filt]
# 罫線に切り取り範囲が被らないようにランダムに切り取りサイズを設定
crop_area_upper_left = [
    (random.randint(x-left,x+min(right-left,0)),random.randint(y-up,y+min(down-up,0))) for left,right,up,down,x,y in dist_from_border]

# 画像の切り取り
x_crop = np.array([x[contours_flatten_page_index_filt[n],u:u+one_page_size,l:l+one_page_size] for n,(l,u) in enumerate(crop_area_upper_left)])
y_crop = np.array([y[contours_flatten_page_index_filt[n],u:u+one_page_size,l:l+one_page_size] for n,(l,u) in enumerate(crop_area_upper_left)])
crop_count = x_crop.shape[0]

print("create {} contours".format(crop_count))

# save matrix
os.makedirs("../temp",exist_ok=True)
np.save("../temp/x_crop",x_crop)
np.save("../temp/y_crop",y_crop)