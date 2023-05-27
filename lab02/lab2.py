import sys
import cv2 as cv
import numpy as np



def global_method(img):
    print("Choose a threshold between 0 and 255: ")
    T = np.uint8(input())

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    newimg = []
    size = img.shape[0]
    for i in range(size):
        a = []
        for j in range(size):
            if(img[i][j] > T):
                a.append(np.uint8(255))
            else:
                a.append(np.uint8(0))
        newimg.append(a)
    newimg = np.array(newimg)

    return newimg



def otsu_method(img):
    print("Still in development!")

    # Stub
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    size = img.shape[0]
    # Calculate the average intensity of the image
    avg = 0;
    for i in range(size):
        for j in range(size):
            avg = avg + int(img[i][j]/(size*size))
    print(avg)

    # Calculate the total variance

    # Calculate the ratio n(T) for every possible T (0-255)

    # Choose the highest T as the threshold

    # Apply that threshold to the image
    newimg = []
    for i in range(size):
        a = []
        for j in range(size):
            if(img[i][j] > T):
                a.append(np.uint8(255))
            else:
                a.append(np.uint8(0))
        newimg.append(a)
    newimg = np.array(newimg)

    return newimg



def main(argv):
    img_path = argv[1]
    op = int(argv[2])
    error_warning = "Invalid operation, please choose a number between 1 and 9."
    window_name = "lab2"
    output_name = "output.png"

    img = cv.imread(img_path)
    if op == 1:
        img = global_method(img)
    elif op == 2:
        img = otsu_method(img)
    else:
        print(error_warning)
        return 0

    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.imwrite(output_name, img)



if __name__ == "__main__":
    main(sys.argv)
