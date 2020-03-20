import cv2
import pyros as ros
import numpy as np
from ORB import ORB
from ColorCorrection import color

def main():
    # Create a VideoCapture object
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("videos/street 3.mp4")

    derecha = cv2.imread("derecha.jpg")
    izquierda = cv2.imread("izquierda.jpg")

    # Check if camera opened successfully

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_width, frame_height)

    cont = 0
    distancia = 2
    dist = distancia
    cdist = 0
    img_list = []
    match = None
    this = None


    while (True):
        ret, frame = cap.read()


        if ret == True:
            # Read display in gs
            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #clr = color(frame)
            #img = clr.correct
            if cont == 0:
                img_list.append(frame)
            elif cont <= cap.get(cv2.CAP_PROP_FRAME_COUNT) and cont == dist:
                if len(img_list) == 1:
                    imA = img_list.pop()
                    # imB = frame
                    #                     img_list.append(imA)
                    #                     img_list.append(imB)
                    #                     orb = ORB(imB)
                    #                     this = orb.stitch([imB, imA], 3)
                # else:
                imB = frame
                img_list.append(imA)
                cv2.imwrite('A this.jpg', imA)
                img_list.append(imB)
                orb = ORB(imB)
                if cdist % 2 == 3:
                    this = orb.stitch([imB, imA], 0, cdist)
                else:
                    this = orb.stitch([imA, imB], 1, cdist)
                print("stitched ", cont)

                if len(img_list) > 1:
                    img_list.pop()
                    img_list.pop()
                    img_list.append(this)
                else:
                    img_list.pop()
                    img_list.append(this)
                cdist += 1
                dist += distancia

                match = this
                imA = this
                # cv2.imshow("match result", match)
                print("shape 0 ", match.shape[0], "shape 1", match.shape[1])

            # Display the resulting frame
            cv2.imshow("submarine", frame)
            cont += 1

            #'q' para detener el video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #fin del while

        else:
            break

        #termina el video

    status = cv2.imwrite("stitched.jpg", match)
    print("saved: ", status)
    cap.release()

    #cierra la ventana
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()