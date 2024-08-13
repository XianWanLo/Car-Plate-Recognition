import cv2 as cv
import numpy as np
import glob
import os.path as osp
from useful_functions import  char_segmentation,resize_img
from char_recognization import template_matching,accuracy


images_list = glob.glob("resources/images/easy/*.jpg")

answer = [['沪', 'E', 'W', 'M', '9', '5', '7'], \
          ['沪', 'A', 'F', '0', '2', '9', '7', '6'], \
          ['鲁', 'N', 'B', 'K', '2', '6', '8']] 

color_range = np.array(([[100,150,50],[140,255,255]],    # blue
                        [[75,43,46],[100,255,255]]),     # green
                        dtype="uint8")  

color_list = ("blue","green")
result_list = []

i=0

for fname in images_list:

    img = cv.imread(fname)
    img_resized = resize_img(img,400)

    mask_area_list = []
    mask_list = []  

    gray = cv.cvtColor(img_resized,cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    for num in range(2):
        
        # 颜色识别
        mask = cv.inRange(hsv, lowerb = color_range[num,0], upperb = color_range[num,1])
        cv.imshow('mask'+str(num),mask)
        mask_list.append(mask)
        mask_area_list.append((mask==255).sum())
    
    ## 车牌为蓝色
    if mask_area_list[0] > mask_area_list[1] :  
               
            _,binary = cv.threshold(gray,127,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
            cv.imshow('binary',binary)
    
     ## 车牌为绿色
    else:           
            _,binary = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV| cv.THRESH_OTSU)
            cv.imshow('binary',binary)


    # 开运算 
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2)) 
    morph1 = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)

    #腐蚀
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,1)) 
    morph2 = cv.erode(morph1,kernel)

    #膨胀
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,5)) 
    morph3 = cv.dilate(morph2,kernel)
    cv.imshow('morph3',morph3)

    word_images = char_segmentation(morph3)
    result = template_matching(word_images)
    
    print(fname)
    print(result)
   
    cv.waitKey(0)    

    accuracy(result,answer[i])

    i+=1



