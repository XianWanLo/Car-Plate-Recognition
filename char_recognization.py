import cv2 as cv
import numpy as np
import glob
import os


template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', 
            '鲁', '蒙', '闽','宁','青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

# 中文模板列表（只匹配车牌的第一个字符）
def get_chinese_words_list():
    chinese_words_list = []
    for i in range(34,64):
        c_word = read_directory('./template/'+ template[i])
        chinese_words_list.append(c_word)
    return chinese_words_list
chinese_words_list = get_chinese_words_list()

# 英文模板列表（只匹配车牌的第二个字符）
def get_num_words_list():
    num_words_list = []
    for i in range(0,10):
        n_word = read_directory('./template/'+ template[i])
        num_words_list.append(n_word)
    return num_words_list
num_words_list = get_num_words_list()


# 英文数字模板列表（匹配车牌后面的字符）
def get_eng_num_words_list():
    eng_num_words_list = []
    for i in range(0,34):
        word = read_directory('./template/'+ template[i])
        eng_num_words_list.append(word)
    return eng_num_words_list
eng_num_words_list = get_eng_num_words_list()


# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
    template_img=cv.imdecode(np.fromfile(template,dtype=np.uint8),1)
    template_img = cv.cvtColor(template_img, cv.COLOR_RGB2GRAY)
    ret, template_img = cv.threshold(template_img, 0, 255, cv.THRESH_OTSU)
    image_ = image.copy()
    height, width = image_.shape
    template_img = cv.resize(template_img, (width, height))
    result = cv.matchTemplate(image_, template_img, cv.TM_CCOEFF)#相关系数匹配，返回值愈大，匹配值越高
    return result[0][0]


def template_matching(word_images):
    results = []
    for index,word_image in enumerate(word_images):
        if index == 0:
            best_score = []
            for chinese_words in chinese_words_list:
                score = []
                for chinese_word in chinese_words:
                    result = template_score(chinese_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[34+i])
            r = template[34+i]
            results.append(r)
            continue

        if index>=1 and index <=4:

            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                for eng_num_word in eng_num_word_list:
                    result = template_score(eng_num_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[i])
            r = template[i]
            results.append(r)
            continue

        else:
            best_score = []
            for num_word_list in num_words_list:
                score = []
                for num_word in num_word_list:
                    result = template_score(num_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[i])
            r = template[i]
            results.append(r)
            continue

    return results


def accuracy(predict,answer):

    correct = 0
    total = len(answer)

    for i in range(len(answer)):

        if answer[i] == predict[i]:
            correct += 1
    
    acc = (correct/total)*100
    print("Prediction Accuracy = %f " % (acc))

    return(acc)