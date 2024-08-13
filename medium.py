import cv2 as cv
import numpy as np
import glob
import os.path as osp
from useful_functions import find_car_plate,resize_img,remove_unwanted_component,char_segmentation
from char_recognization import template_matching,accuracy


images_list = glob.glob("resources/images/medium/*.jpg")

color_range = np.array([[100,150,50],[140,255,255]])    # blue
                       

answer = [['沪', 'E', 'W', 'M', '9', '5', '7'], \
          ['豫', 'B', '2', '0', 'E', '6', '8'],\
          ['沪', 'A', '9', '3', 'S', '2', '0'] ]

i=0

for fname in images_list:

    img = cv.imread(fname)

    img_resized = resize_img(img,1000)

    gray = cv.cvtColor(img_resized,cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lowerb = color_range[0], upperb = color_range[1])       
    _,binary = cv.threshold(mask,0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
    cv.imshow('binary',binary)

    car_plate = find_car_plate(img_resized,binary)
    new_carplate = remove_unwanted_component(car_plate)
    cv.imshow('new_carplate',new_carplate)

    # 形态学操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,2))  #去除横向细线
    morph1 = cv.morphologyEx(new_carplate,cv.MORPH_CLOSE,kernel)
    cv.imshow('morph1',morph1)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,1))  #去除竖向细线
    morph2 = cv.morphologyEx(morph1,cv.MORPH_OPEN,kernel)
    cv.imshow('morph2',morph2)

    word_images = char_segmentation(morph2)
    result = template_matching(word_images)
    print(fname)
    print(result)

    accuracy(result,answer[i])
    
    cv.waitKey(0)  

    i+=1     

