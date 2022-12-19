import os
import cv2
import numpy as np
import math
import glob

#app_win_size_x = 870
#app_win_size_y = 500
γ_range_upper_bound = 5

output_dir_path = output_RGB_path = output_HSI_path = output_LAB_path = ""

@staticmethod
def root_path(): #當前 working dir 之 root path
    return os.getcwd()
    #return "/workspaces/mvl/ImageProcessing"

def set_output_path():
    global output_dir_path, output_RGB_path, output_HSI_path, output_LAB_path
    output_dir_path = os.path.join(root_path(), "hw3_output")
    output_RGB_path = os.path.join(output_dir_path, "RGB")
    output_HSI_path = os.path.join(output_dir_path, "HSI")
    output_LAB_path = os.path.join(output_dir_path, "LAB")
    
def get_output_path():
    global output_dir_path, output_RGB_path, output_HSI_path, output_LAB_path
    return [output_dir_path, output_RGB_path, output_HSI_path, output_LAB_path]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("mkdir "+dir_path)    
    else:
        print(dir_path+" already exist, no need to mkdir.")

def get_image_path(path): #root_path/HW2_test_image
    return glob.glob(os.path.join(path, "*.bmp"))+glob.glob(os.path.join(path, "*.tif"))+glob.glob(os.path.join(path, "*.jpg"))

def show_img_fullscreen(img_name, showimg ,type):
    cv2.namedWindow(img_name, type)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(img_name, app_win_size_x,app_win_size_y)
    #cv2.moveWindow(img_name, app_pos_x,app_pos_y)
    cv2.imshow(img_name, showimg)

def read_and_operate_image(image_path):
    image =cv2.imread(image_path)
    #show_img_fullscreen("Current Image: "+image_path, image, cv2.WINDOW_KEEPRATIO)
    image_gray =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #show_img_fullscreen("Current Image(grayscale): "+image_path, image_gray, cv2.WINDOW_KEEPRATIO)

    image_RGB = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    return image, image_gray, image_RGB

#algorithm implementation
#power-law
"""-------------------------------"""
def power_law_transform(img, γ):
    if isinstance(img, np.ndarray) and isinstance(γ, float):
        return np.array(255*(img/255)**γ, dtype='uint8')
    else:
        print("img: "+str(type(img)),end=" ")
        print("γ: "+str(type(γ)))
        print("type error.")
        exit()

def choose_pl(img, range_upper_bound):
    record = 999
    value = 0
    range_upper_bound *=10
    best = img
    for γ in range(0, range_upper_bound ,1):
        temp_img = power_law_transform(img, γ/10)
        cal = abs(np.mean(temp_img)-128)
        #print(γ/10, end=": ")
        #print(cal)
        if  cal < record:
            record = cal
            best = temp_img
            value = γ/10

    print("best record=",end="")
    print(record,end=",")
    print(" γ = ", end="")
    print(value)

    return best

def choose_pl_hsi(img):
    record = 999
    value = 0
    best = img
    γhsi=[0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    for index in range(0,6):
        temp_img = power_law_hsi(img_HSI, γhsi[index])
        temp_img = hsitorgb(temp_img)
        cal = abs(np.mean(temp_img)-128)
        if  cal < record:
            record = cal
            best = temp_img
            value = γhsi[index]
    print("best record=",end="")
    print(record,end=",")
    print(" γ = ", end="")
    print(value)

    return best

def power_law_hsi(image, gamma):
    start = 1   
    for i in range(start,3):
        image[:,:,i] = image[:,:,i].astype(np.float32)
        image[:,:,i] = (image[:,:,i]/255.)**gamma *255. # s=cr^gamma
        image[:,:,i] = image[:,:,i].astype(np.uint8)
    
    return image
#Histogram Equalization
"""-------------------------------
先把所有顏色的數量都統計成直方圖，並計算每個顏色的機率(像素個數/像素總數)。
再計算累計機率，乘上像素總數並4捨5入得到新的像素值。
-------------------------------"""
def histogram_equalization(img):
    hist, bins= np.histogram(img.ravel(), 256, [0, 255])    #直方圖 灰階:256bins，0 ~ 255
    pdf = hist/img.size   #計算每個顏色機率
    cdf = pdf.cumsum()
    equ_value = np.round((cdf*255)).astype('uint8')
    
    return equ_value[img]

def rgbtohsi(rgb_Img):
    [rows, cols] = [int(rgb_Img.shape[0]), int(rgb_Img.shape[1])]
    H, S, I = B, G, R = cv2.split(rgb_Img)
    # nomoralize to [0,1]
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0
    hsi_Img = rgb_Img.copy()
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif B[i, j] <= G[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(B[i, j], G[i, j]), R[i, j])
            sum = B[i, j] + G[i, j] + R[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            
            hsi_Img[i, j, 0] = H * 255
            hsi_Img[i, j, 1] = S * 255
            hsi_Img[i, j, 2] = I * 255

    return hsi_Img

def hsitorgb(hsi_img):
    [h,w] = [int(hsi_img.shape[0]), int(hsi_img.shape[1])]
    B, G, R = H, S, I = cv2.split(hsi_img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = B * 255
            bgr_img[i, j, 1] = G * 255
            bgr_img[i, j, 2] = R * 255
    return bgr_img

def hsi_fun(images, image_path, image_dir, hsi):
    img_HSI = rgbtohsi(images[0])
    fn=image_path[0].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    ehsi = power_law_hsi(img_HSI, 0.6)
    ehsi = hsitorgb(ehsi)
    cv2.imwrite(fn.replace(image_dir, hsi)+".bmp", ehsi)

    img_HSI = rgbtohsi(images[1])
    ehsi = power_law_hsi(img_HSI, 1.2)
    ehsi = hsitorgb(ehsi)
    fn=image_path[1].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    cv2.imwrite(fn.replace(image_dir, hsi)+".bmp", ehsi)

    img_HSI = rgbtohsi(images[2])
    ehsi = power_law_hsi(img_HSI, 1.2)
    ehsi = hsitorgb(ehsi)
    fn=image_path[2].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    cv2.imwrite(fn.replace(image_dir, hsi)+".bmp", ehsi)

    img_HSI = rgbtohsi(images[3])
    ehsi = power_law_hsi(img_HSI, 0.8)
    ehsi = hsitorgb(ehsi)
    fn=image_path[3].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    cv2.imwrite(fn.replace(image_dir, hsi)+"_light.bmp", ehsi)
    img_HSI = rgbtohsi(images[3])
    ehsi = power_law_hsi(img_HSI, 1.0)
    ehsi = hsitorgb(ehsi)
    cv2.imwrite(fn.replace(image_dir, hsi)+"_dark.bmp", ehsi)

def lab_fun(images, image_path, image_dir, lab):
    print("labing")
    [γ0,γ1,γ2,γ3] = [0.9,1.1,0.9,0.9]
    fn=image_path[0].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    img_lab = cv2.cvtColor(images[0], cv2.COLOR_RGB2Lab)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-1.bmp", img_lab)
    elab = power_law_transform(img_lab, γ0)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-2.bmp", elab)
    elab = cv2.cvtColor(elab, cv2.COLOR_Lab2RGB)
    cv2.imwrite(fn.replace(image_dir, lab)+"("+str(γ0)+").bmp", elab)

    fn=image_path[1].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    img_lab = cv2.cvtColor(images[1], cv2.COLOR_RGB2Lab)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-1.bmp", img_lab)
    elab = power_law_transform(img_lab, γ1)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-2.bmp", elab)
    elab = cv2.cvtColor(elab, cv2.COLOR_Lab2RGB)
    cv2.imwrite(fn.replace(image_dir, lab)+"("+str(γ1)+").bmp", elab)

    fn=image_path[2].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    img_lab = cv2.cvtColor(images[2], cv2.COLOR_RGB2Lab)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-1.bmp", img_lab)
    elab = power_law_transform(img_lab, γ2)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-2.bmp", elab)
    elab = cv2.cvtColor(elab, cv2.COLOR_Lab2RGB)
    cv2.imwrite(fn.replace(image_dir, lab)+"("+str(γ2)+").bmp", elab)

    fn=image_path[3].replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    img_lab = cv2.cvtColor(images[3], cv2.COLOR_RGB2Lab)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-1.bmp", img_lab)
    elab = power_law_transform(img_lab, γ3)
    #cv2.imwrite(fn.replace(image_dir, lab)+"-2.bmp", elab)
    elab = cv2.cvtColor(elab, cv2.COLOR_Lab2RGB)
    cv2.imwrite(fn.replace(image_dir, lab)+"("+str(γ3)+").bmp", elab)

# main
image_dir = os.path.join(root_path(), "HW3_test_image")
print(image_dir)
images = get_image_path(image_dir)  #取得圖片路徑
print(images)

set_output_path()
for output_path in get_output_path():
    mkdir(output_path)

dir, rgb, hsi, lab = get_output_path()
imgrgbset=[]

for image in images:
    file = image.replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    img, img_gray, img_RGB= read_and_operate_image(image)
    imgrgbset.append(img_RGB)

    img_pl = choose_pl(img_RGB, γ_range_upper_bound)
    img_he = histogram_equalization(img_RGB)
    cv2.imwrite(file.replace(image_dir, rgb)+"_pl.bmp", img_pl)
    cv2.imwrite(file.replace(image_dir, rgb)+"_he.bmp", img_he)

hsi_fun(imgrgbset, images, image_dir, hsi)
lab_fun(imgrgbset, images, image_dir, lab)