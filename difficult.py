import cv2 as cv
import numpy as np
import glob
import os.path as osp
from useful_functions import find_car_plate,resize_img,remove_unwanted_component,char_segmentation
from char_recognization import template_matching,accuracy


images_list = glob.glob("resources/images/difficult/*.jpg")

answer = [['沪', 'E', 'W', 'M', '9', '5', '7'], \
         ['沪', 'A', 'D', 'E', '6', '5', '9', '8'], \
         ['皖', 'S', 'J', '6', 'M', '0', '7']]

color_range = np.array(([[100,150,50],[140,255,255]],    # blue
                        [[55,50,50],[100,255,255]]),     # green
                        dtype="uint8")  

i=0

for fname in images_list:

    img = cv.imread(fname)

    img_resized = resize_img(img,1000)

    mask_area_list = []
    mask_list = []  

    gray = cv.cvtColor(img_resized,cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    for num in range(2):
           
        mask = cv.inRange(hsv, lowerb = color_range[num,0], upperb = color_range[num,1])       
        mask_list.append(mask)
        mask_area_list.append((mask==255).sum())
    
    if mask_area_list[0] > mask_area_list[1] :  ## 车牌为蓝色              
            flag = 'blue'
            _,binary = cv.threshold(mask_list[0],0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
            #cv.imshow('MASK',mask_list[0])
    else:             
            flag = "green"
            _,binary = cv.threshold(mask_list[1],0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
            #cv.imshow('MASK',mask_list[1])

    car_plate = find_car_plate(img_resized,binary,difficult = True,flag = flag)
    new_carplate = remove_unwanted_component(car_plate)

    low_w = int(0.02*new_carplate.shape[1])
    high_w = int(0.965*new_carplate.shape[1])
    new_carplate =  new_carplate[:,low_w:high_w]
    new_carplate[0,:] = 0
    new_carplate[-1,:] = 0

    cv.imshow('car_plate',new_carplate)

    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (2,2))  #去除小白点
    kernelY = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    image = cv.dilate(new_carplate, kernelX)  #膨胀
    image = cv.erode(image, kernelY)   #腐蚀

    cv.imshow('image',image)

    word_images = char_segmentation(image)
    result = template_matching(word_images)
    print(fname)
    print(result)
    cv.waitKey(0)  

    accuracy(result,answer[i])   

    i+=1  

