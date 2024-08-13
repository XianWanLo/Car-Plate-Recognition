import cv2 as cv
import numpy as np
import glob
import os.path as osp
from skimage.filters import threshold_local


def order_points(pts):
	
	rect = np.zeros((4, 2), dtype = "float32")
	
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
        
	# compute the perspective transform matrix and then apply it
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def resize_img(img, max_size):
    """ resize图像 """
    h, w = img.shape[0:2]
    scale = max_size / max(h, w)
    img_resized = cv.resize(img, None, fx=scale, fy=scale, 
                            interpolation=cv.INTER_CUBIC)
    return img_resized


def find_rectangle(contour):
    """ 寻找矩形轮廓 """
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks



def find_car_plate(image,binary,difficult= False,flag = "blue"):

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_cont = image.copy()
    img_cont = cv.drawContours(img_cont, contours, -1, (0, 0,255), 6)
    cv.imshow("Contours", img_cont)
    
    block = []

    for contour in contours:
            
            r = find_rectangle(contour)    # 轮廓的左上点和右下点
            a = (r[2] - r[0]) * (r[3] - r[1])   # 面积
            s = (r[2] - r[0]) / (r[3] - r[1])   # 长度比
            block.append([r, a, s])

    # bounded car plate on original img       
    block = sorted(block, key=lambda bl: bl[1])[-1:]
    rect = block[0][0]

    img_cont2 = image.copy()
    cv.rectangle(img_cont2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 6)
    cv.imshow('bounded_box',img_cont2)

    # crop car plate
    if difficult == False:

        img_cont3 = image.copy()
        license_img = img_cont3[rect[1]:rect[3], rect[0]:rect[2]]
        cv.imshow('license_img',license_img)

        license_bi = binary[rect[1]:rect[3], rect[0]:rect[2]]
        license = cv.bitwise_not(license_bi)
        license = resize_img(license,400)
        cv.imshow('license_binary',license)

    else: 
        
        # 二次精准定位
        img_cont3 = image.copy()
        license_exp = img_cont3[rect[1]-40:rect[3]+40, rect[0]-50:rect[2]+50]
        license_exp = resize_img(license_exp,400)
        cv.imshow('license_expand',license_exp)

        license_bi = binary[rect[1]-40:rect[3]+40, rect[0]-50:rect[2]+50]
        license_bi = resize_img(license_bi,400)

        if flag =="blue":
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
            license = cv.erode(license_bi, kernel)  #腐蚀

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (10,20))
            license = cv.dilate(license, kernel,iterations=2)  #膨胀
            cv.imshow('license_blue',license)

        if flag == "green":
            gray = cv.cvtColor(license_exp,cv.COLOR_BGR2GRAY)
            _,binary = cv.threshold(gray,40,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,3))
            binary = cv.dilate(binary, kernel)  #膨胀

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (8,15))
            binary = cv.erode(binary, kernel)  #腐蚀

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,30))
            binary = cv.dilate(binary, kernel)  #膨胀
            license = binary
            cv.imshow('license_green',license)

        _,license = cv.threshold(license,127,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
        cv.imshow('license_first',license)

        contours, hierarchy = cv.findContours(license.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        license_1 = license_exp.copy()
        license_1 = cv.drawContours(license_1, contours, -1, (255, 0, 0), 6)

        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        license_2 = license_exp.copy()
        license_2 = cv.drawContours(license_2, cnt, -1, (255, 0, 0), 6)
        cv.imshow('license_second',license_2)   
        
        peri = cv.arcLength(cnt, True) 
        approx = cv.approxPolyDP(cnt,0.02*peri,True)
        print(len(approx))
	        
        if len(approx) == 4:
            screenCnt = approx
            
        license_3 = license_exp.copy()
        warped = four_point_transform(license_3, screenCnt.reshape(4, 2) * 1)   
        gray = cv.cvtColor(warped,cv.COLOR_BGR2GRAY) 
        
        if flag =="blue":
            _,binary = cv.threshold(gray,150,255,cv.THRESH_BINARY| cv.THRESH_OTSU)    

        if flag =="green":
            _,binary = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV| cv.THRESH_OTSU)    
            
        license = resize_img(binary,400)

        cv.imshow("warped", warped)
        cv.imshow("license", license)

    return license 


def remove_unwanted_component(car_plate):

    #图片的像素直方图
    row_histogram = np.sum(car_plate, axis=1)    
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / car_plate.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)

    # 挑选跨度最大的波峰
    wave_span = 0.0
    selected_wave = []
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    new_carplate = car_plate[selected_wave[0]:selected_wave[1], :]

    return new_carplate


def find_end(start, black, width, black_max):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m]) > (0.95*black_max):
            end = m
            break
    return end


def char_segmentation(thresh):
    """ 分割字符 """
    white, black = [], []    # list记录每一列的黑/白色像素总和
    height, width = thresh.shape
    
    white_max = 0    # 仅保存每列，取列中白色最多的像素总数
    black_max = 0    # 仅保存每列，取列中黑色最多的像素总数
    # 计算每一列的黑白像素总和
    for i in range(width):
        line_white = 0    # 这一列白色总数
        line_black = 0    # 这一列黑色总数
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)

    word_images = []

    # 分割车牌字符char_segmentation
    n = 1
    while n < width - 2:
        n += 1
        #黑底白字
        if (white[n]) > (0.05 * white_max):  
            start = n
            end = find_end(start, black, width, black_max)
            n = end
            if end - start > 12 or end > (width * 3 / 7):
                cropImg = thresh[0:height, start-1:end+1]
                # 对分割出的数字、字母进行S保存
                cropImg = cv.resize(cropImg, (51, 84))
                word_images.append(cropImg)
                cv.imshow('char{}'.format(n), cropImg)

    return word_images
