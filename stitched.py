import cv2
import pyros as ros
import numpy as np
import itertools
from operator import itemgetter
from ORB import ORB
from ColorCorrection import color

def main():
    # Create a VideoCapture object
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("videos/AMBA0136.mp4")

    #derecha = cv2.imread("derecha.jpg")
    #izquierda = cv2.imread("izquierda.jpg")

    # Check if camera opened successfully

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #print(frame_width, frame_height)

    cont = 0
    img_list = []

    while (True):
        ret, frame = cap.read()

        if ret == True:
            # Read display in gs
            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('video', frame)

            #clr = color(frame)
            #img = clr.correct

            img_list.append(frame)

            #'q' para detener el video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #fin del while
        else:
            break

        #termina el video

    #if cont == 0:
        # orb = ORB(img, frame)
        #orb = ORB(frame)
        #match = orb
        #cont += 1
    #elif cont > 0:
        # orb = ORB(img, frame)
        #orb = ORB(frame)
        #match = orb.stitch([orb.img, match.img])
        #status = cv2.imwrite("stitched_lab.jpg", match)
        #print("saved: ", status)
        #cv2.imshow("match result", match)
        #cont += 1

    my_list = range(0, len(img_list))
    stitcher = ORB(frame)

    # Go over all pairs to find a correlation in the correct order
    for pair in itertools.combinations(my_list, r=2):
        imA = itemgetter(itemgetter(1)(pair))(img_list)
        imB = itemgetter(itemgetter(0)(pair))(img_list)

        # Stitch th pair together
        (this) = stitcher.stitch([imA, imB])

        # If a result exists then remove them from the image list and put the stitched image in
        if this.any():
            if len(img_list) > 1:
                img_list.pop(itemgetter(0)(pair))
                img_list.insert(0, this)
                img_list.pop(itemgetter(1)(pair))
            else:
                img_list.pop(itemgetter(0)(pair))
                img_list.insert(0, this)
            break

        # Put the final result in a new variable
    result = img_list.pop()
    cv2.imshow("result", result)

    cap.release()

    #cierra la ventana
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()