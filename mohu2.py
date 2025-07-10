# coding:utf8 
import os

from PIL import Image, ImageFilter

filename = os.listdir("./test2/")
base_dir = "./test2/"
new_dir = "./test3/"

for img in filename:
    image = Image.open(base_dir + img)
    image_mohu = image.filter(ImageFilter.BLUR)
    image_mohu.save(new_dir + img)
