import os

image_dir = 'dir'
image_list = os.listdir(image_dir)
o_list = []
for i in image_list:
    if "_O.JPG"  in i :
        o_list.append(i)

for i in o_list:
    os.remove(image_dir+"/"+i)
