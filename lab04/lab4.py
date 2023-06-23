import sys
import cv2 as cv



def rotate(img, t_angle):
    print("Rotation Function is still in development!")
    return img



def scale(img, t_scale, t_im):
    return img



def project(img, t_projection):
    return img



def main(argv):
    ERROR_WARNING = "ERROR: Invalid Command Line Input"
    HELP = "lab4.py [-a rotation_angle] [-i input_image_path] [-m interpolation_method] [-o output_filename] [-p projection] [-s scale_factor]"
    WINDOW_NAME = "lab4.py"

    t_angle = 0
    t_scale = 1.0
    t_im = 0
    t_projection = 0
    image_path = "img/baboon.png"
    output_filename = "example.png"

    for i in range(1, len(argv)):
        if i%2:
            if argv[i] == "-a":
                t_angle = float(argv[i + 1])
            elif argv[i] == "-h":
                print(HELP)
            elif argv[i] == "-i":
                image_path = argv[i + 1]
            elif argv[i] == "-m":
                t_im = int(argv[i + 1])
            elif argv[i] == "-o":
                output_filename = argv[i + 1]
            elif argv[i] == "-p":
                t_projection = int(argv[i + 1])
            elif argv[i] == "-s":
                t_scale = float(argv[i + 1])
            else:
                print(ERROR_WARNING)
                return 0

    img = cv.imread(image_path)
    img = rotate(img, t_angle)
    img = scale(img, t_scale, t_im)
    img = project(img, t_projection)

    cv.imshow(WINDOW_NAME, img)
    cv.waitKey(0)
    cv.imwrite(output_filename, img)

    return 1



if __name__ == "__main__":
    main(sys.argv)
