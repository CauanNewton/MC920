import sys
import cv2 as cv
import numpy as np



def example(img):
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

def main(argv):
    img_path = argv[1]
    op = int(argv[2])
    error_warning = "Invalid Operation"
    window_name = "lab4"
    output_name = "output.png"
    
    img = cv.imread(img_path)
    if op == 1:
    elif op == 2:
    elif op == 3:
    elif op == 4:
    elif op == 5:
    else:
        print(error_warning)
        return 0
        
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.imwrite(output_name, img)



if __name__ == "__main__":
    main(sys.argv)
