import cv2
import numpy as np
from random import randint
import random
from PIL import Image

height = 32
width = height

size = 6

pos_all = [
    (5, 5 ),
    (5, 15),
    (5, 25),
    (15, 5 ),
    (15, 15),
    (15, 25),
    (25, 5 ),
    (25, 15),
    (25, 25),
]

def is_hit(ver1, ver2, fig_type):
    if fig_type == "square" or fig_type == "juji" or fig_type == "triangle":
        if ( abs(ver2[0] - ver1[0]) < size ) and ( abs(ver2[1] - ver1[1]) < size ) :
            return True
    if fig_type == "circle" :
        if ( (ver2[0] - ver1[0]) ** 2.0 + (ver2[1] - ver1[1]) ** 2.0 < (size) ** 2.0 ):
            return True
    return False

def are_hit(ver, dones, fig_type):
    for i in range(len(dones)):
        if is_hit(ver, dones[i], fig_type):
            return True
    return False

def draw(ver, size, fig_type, img, color):
    if fig_type == "square":
        for i in range(ver[0] - int(size/2), ver[0] + int(size/2)):
            for j in range(ver[1] - int(size/2), ver[1] + int(size/2)):
                (img[i,j,0], img[i,j,1], img[i,j,2]) =  color
    elif fig_type == "circle":
        cv2.circle(img, (ver[0],ver[1]), int(size / 2),  color, thickness= -1 )
    elif fig_type == "juji":
        cv2.drawMarker(img, (ver[0],ver[1]),  color , markerType=cv2.MARKER_CROSS, markerSize=size )
    elif fig_type == 'triangle':
        triangle_cnt = np.array( [(ver[0], ver[1]), (ver[0] + size, ver[1]), ( int(ver[0] + size /2), int (ver[1] - size * (3 ** 0.5)/2 ) )] )
        cv2.drawContours(img, [triangle_cnt], 0,  color, -1)



def create_images(num, fig_type, color):
    img = np.zeros((height, width, 3), np.uint8)
    dones = []

    if num == 0:
        return img / 255.0

    #まず、どのブロックに配置するか決める
    block_idx = random.sample(range(9),num)
    for id in block_idx:
        pt = pos_all[id]
        draw(pt, size, fig_type, img, color)
    return img / 255.0

    """ while True:
        if fig_type == 'square' :
            ver = [randint(0,height - size), randint(0, height-size)]
        elif fig_type =='triangle':
            ver = [randint(0,height - size), randint( int (size * (3 ** 0.5)/2) , height )]
        elif fig_type == "circle" or fig_type == "juji" :
            ver = [randint(size/2 ,height - size/2), randint(size/2 , height-size/2)]

        if not are_hit(ver, dones, fig_type):
            draw(ver, size, fig_type, img, color)
            dones.append(ver)
            if len(dones) == num:
                break
    return img / 255.0 """



def save_image(img, fig_type, num):
    cv2.imwrite('images/' + fig_type + str(num) + '.png', img)

#colors = [(255, 0 ,0),(0, 255,0),(0, 0, 255), (255, 255, 255)]


#画像の保存
""" res = create_images(5, 'juji', (255, 0 ,0))
tar = (res * 255).astype(np.uint8)
pil_image = Image.fromarray(tar)
pil_image.save('generated_images/test.png')

imgArray = np.asarray(pil_image) """


""" for fig,c in zip(['square', 'circle','juji','triangle'], colors):
    create_images(9, fig, c) """
