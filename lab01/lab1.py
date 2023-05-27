import sys
import cv2 as cv
import numpy as np



def tile_and_shuffle_image(img):
    tiles = 16
    splits = 4
    size = img.shape[0]
    tile_size = int(size/splits)

    newimg = []
    for i in range(splits):
        for j in range(splits):
            newimg.append(img[tile_size*i:tile_size*(i+1), tile_size*j:tile_size*(j+1)])
    newimg = np.array(newimg)

    return newimg



def combine_images(img1):    
    print("Assign a weight (0-1) to image 1:")
    w1 = float(input())
    print("Write down the path to image 2:")
    img_path2 = input()
    print("Assing a weight (0-1) to image 2:")
    w2 = float(input())
    img2 = cv.imread(img_path2)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
    newimg = []
    size = img1.shape[0]
    for i in range(size):
        a = []
        for j in range(size):
            a.append(np.uint8(img1[i][j]*w1 + img2[i][j]*w2))
        newimg.append(a)
    newimg = np.array(newimg)

    return newimg



def negative_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    newimg = []
    size = img.shape[0]
    for i in range(size):
        a = []
        for j in range(size):
            img[i][j] = 255 - img[i][j]
            a.append(np.uint8(img[i][j]))
        newimg.append(a)        
    newimg = np.array(newimg)
    
    return newimg



def transform_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Choose the minimum color intensity:")
    min = int(input())
    print("Choose the maximum color intensity:")
    max = int(input())
    color_depth = 256
    ratio = (max - min)/color_depth
    
    newimg = []    
    size = img.shape[0]    
    for i in range(size):
        a = []
        for j in range(size):
            img[i][j] = min + img[i][j]*ratio
            a.append(np.uint8(img[i][j]))
        newimg.append(a)        
    newimg = np.array(newimg)
    
    return newimg



def even_rotation(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    newimg = []
    size = img.shape[0]
    for i in range(size):
        if i % 2 == 0:
            newimg.append(img[i][::-1])
        else:
            newimg.append(img[i])        
    newimg = np.array(newimg)
    
    return newimg



def reflect_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    newimg = []
    size = img.shape[0]
    start = int(size/2)
    for i in range(start):
        newimg.append(img[i])
    for i in range(start, size):
        newimg.append(img[size - i])
    newimg = np.array(newimg)
    
    return newimg



def mirror_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    newimg = []
    size = img.shape[0]
    for i in range(size):
        newimg.append(img[size - i - 1])
    newimg = np.array(newimg)
    
    return newimg



def recolor_image(img):
    new_colors = [[0.131, 0.534, 0.272], 
                  [0.168, 0.686, 0.349], 
                  [0.189, 0.769, 0.393]]
    size = img.shape[0]
    
    newimg = []
    for i in range(size):
        a = []
        for j in range(size):
            img[i][j] = np.dot(img[i][j], new_colors)
            a.append(np.uint8(img[i][j]))
        newimg.append(a)
    newimg = np.array(newimg)
    
    return newimg



def recolor_image_to_mono(img):
    weights = [0.1140, 0.5870, 0.2989]
    size = img.shape[0]
    
    newimg = []
    for i in range(size):
        a = []
        for j in range(size):
            temp = np.dot(img[i][j], weights)
            a.append(np.uint8(temp))
        newimg.append(a)
    newimg = np.array(newimg)
    
    return newimg



def gamma_correction(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Choose a new gamma value:")
    gamma = float(input())
    color_depth = 256
    
    newimg = []
    size = img.shape[0]
    for i in range(size):
        a = []
        for j in range(size):
            temp = img[i][j]/color_depth
            temp = temp**(1/gamma)
            temp = temp*color_depth
            a.append(np.uint8(temp))
        newimg.append(a)
    newimg = np.array(newimg)
    
    return newimg  



def quantize_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Choose a new bit depth value:")
    new_depth = int(input())
    color_depth = 256
    ratio = (new_depth - 1)/(color_depth - 1)
    
    newimg = []
    size = img.shape[0]    
    for i in range(size):
        a = []
        for j in range(size):
            a.append(np.uint8(round(img[i][j]*ratio)*(1/ratio)))
        newimg.append(a)        
    newimg = np.array(newimg)
    
    return newimg



def bit_plane(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Choose which bit plane (0-7) you want to see:")
    plane = int(input())
    
    newimg = []
    size = img.shape[0]
    for i in range(size):
        a = []
        for j in range(size):
            a.append(bin(img[i][j]))
        newimg.append(a)

    return newimg



def filter_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Choose which filter (1-11) to apply:")
    f = int(input())
    kernel = []
    if f == 1:
        kernel = [[ 0,  0, -1,  0,  0],
                  [ 0, -1, -2, -1,  0],
                  [-1, -2, 16, -2, -1],
                  [ 0, -1, -2, -1,  0],
                  [ 0,  0, -1,  0,  0]]
    elif f == 2:
        kernel = [[ 1,  4,  6,  4,  1],
                  [ 4, 16, 24, 16,  4],
                  [ 6, 24, 36, 24,  6],
                  [ 4, 16, 24, 16,  4],
                  [ 1,  4,  6,  4,  1]]
        kernel = np.multiply(kernel, (1/256))
    elif f == 3:
        kernel = [[-1,  0,  1],
                  [-2,  0,  2],
                  [-1,  0,  1]]
    elif f == 4:
        kernel = [[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]]
    elif f == 5:
        kernel = [[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]
    elif f == 6:
        kernel = [[ 1,  1,  1],
                  [ 1,  1,  1],
                  [ 1,  1,  1]]
        kernel = np.multiply(kernel, (1/9))
    elif f == 7:
        kernel = [[-1, -1,  2],
                  [-1,  2, -1],
                  [ 2, -1, -1]]
    elif f == 8:
        kernel = [[ 2, -1, -1],
                  [-1,  2, -1],
                  [-1, -1,  2]]
    elif f == 9:
        kernel = np.identity(9)
        kernel = np.multiply(kernel, (1/9))
    elif f == 10:
        kernel = [[-1, -1, -1, -1, -1],
                  [-1,  2,  2,  2, -1],
                  [-1,  2,  8,  2, -1],
                  [-1,  2,  2,  2, -1],
                  [-1, -1, -1, -1, -1]]
        kernel = np.multiply(kernel, (1/8))
    elif f == 11:
        kernel = [[-1, -1,  0],
                  [-1,  0,  1],
                  [ 0,  1,  1]]
    else:
        print("Invalid choice")
        return 0
    
    kernel = np.array(kernel)    
    newimg = cv.filter2D(img, -1, kernel)
    return newimg



def special_filter(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel1 =    [[-1,  0,  1],
                  [-2,  0,  2],
                  [-1,  0,  1]]
    kernel1 = np.array(kernel1)
    kernel2 =    [[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]]
    kernel2 = np.array(kernel2)
    
    newimg1 = cv.filter2D(img, -1, kernel1)
    newimg2 = cv.filter2D(img, -1, kernel2)
    
    newimg = []
    size = img.shape[0]
    for i in range(size):
        a = []
        for j in range(size):
            a.append(np.uint8((newimg1[i][j]**2 + newimg2[i][j]**2)**0.5))
        newimg.append(a)
    newimg = np.array(newimg)
    
    return newimg



def main(argv):
    img_path = argv[1]
    op = int(argv[2])
    error_warning = "Invalid Operation"
    window_name = "lab1"
    output_name = "output.png"
    
    img = cv.imread(img_path)
    if op == 1:
        print("Sorry, code is still in development!")
        print("No transformation has been applied.")
    elif op == 2:
        img = combine_images(img)
    elif op == 3:
        img = negative_image(img)
    elif op == 4:
        img = transform_image(img)
    elif op == 5:
        img = even_rotation(img)
    elif op == 6:
        img = reflect_image(img)
    elif op == 7:
        img = mirror_image(img)
    elif op == 8:
        img = recolor_image(img)
    elif op == 9:
        img = recolor_image_to_mono(img)
    elif op == 10:
        img = gamma_correction(img)
    elif op == 11:
        img = quantize_image(img)
    elif op == 12:
        print("Sorry, code is still in development!")
        print("No transformation has been applied.")
    elif op == 13:
        img = filter_image(img)
    elif op == 14:
        img = special_filter(img)
    else:
        print(error_warning)
        return 0
        
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.imwrite(output_name, img)



if __name__ == "__main__":
    main(sys.argv)
