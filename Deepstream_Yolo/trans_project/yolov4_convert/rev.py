from PIL import Image
import sys
img = Image.open(sys.argv[1])
img = img.resize([200,200])
img = img.convert("RGBA")  # 转换获取信息
pixdata = img.load()

for y in range(img.size[1]):
    for x in range(img.size[0]):
        
        if pixdata[x, y][0] > 250 and pixdata[x, y][1] > 250 and pixdata[x, y][2] > 200 and pixdata[x, y][3] > 250:
            pixdata[x, y] = (255, 255, 255, 0) #透明

img.save(r".\image\timg_new.png")


# import os
# import paddlehub

# # 加载模型
# humanseg = paddlehub.Module(name="deeplabv3p_xception65_humanseg")

# # 指定抠图图片目录
# path = './image/'
# files = []
# dirs = os.listdir(path)
# for diretion in dirs:
    # files.append(path + diretion)

# # 批量抠图
# results = humanseg.segmentation(data={"image": files})
