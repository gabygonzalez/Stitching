import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter

class ORB():

    def __init__(self, img):
        self.img = img
        self.descriptor = None
        self.keypoints = self.orb(img)

    def orb(self, img):
        orb = cv2.ORB_create()

        keypoints, descriptor = orb.detectAndCompute(img, None)

        self.descriptor = descriptor

        return keypoints

    def crop(self, img, tol=0):
         # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #          _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #
    #          contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #          cnt = contours[0]
    #          x, y, w, h = cv2.boundingRect(cnt)
    #
    #          print(x, y, w, h)
    #          if w >= self.img.shape[1] and h >= self.img.shape[0]:
    #             print("mayor igual w h")
    #             crop = img[y:y + h, x:x + w]
    #          else:
    #              print("x y")
    #              crop = img[y, x]
    #          return crop
        mask = img[:, :, 1] > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def crop_right(self, img):
        height, width = img.shape[:2]

        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        borders = (0, 0, int(width*.8), height)  # .4
        cropped = pil_im.crop(borders)
        cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)

        return cropped

    def crop_mid(self, img):
        height, width = img.shape[:2]

        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        borders = (width*.25, 0, width*.75, height)
        cropped = pil_im.crop(borders)
        cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        return cropped

    def crop_left(self, img):
        height, width = img.shape[:2]

        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        borders = (int(width*.3), 0, width, height) # .6
        cropped = pil_im.crop(borders)
        cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        return cropped

    def unsharp_mask(self, img, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask. """
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharpened = float(amount + 1) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(img - blurred) < threshold
            np.copyto(sharpened, img, where=low_contrast_mask)

        return sharpened

    def stitch(self, images, flag, cont, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images
        # CALIBRAR IMAGEN: BUSCAR CONFIGURACIÓN DE LA CÁMARA CON LA QUE SE GRABÓ

        if flag == 0:
            print("cero")
            imageA = cv2.resize(imageA, (int(imageA.shape[1] * .5), int(imageA.shape[0] * .5)),
                                interpolation=cv2.INTER_AREA)
            imageB = cv2.resize(imageB, (int(imageB.shape[1] * .5), int(imageB.shape[0] * .5)),
                                interpolation=cv2.INTER_AREA)


        (result) = self.stitch_images(imageA, imageB, ratio, reprojThresh)
        #result = cv2.GaussianBlur(result, (3, 3), 1)


        #         if cont%2 == 0:
        #             print("sharpened")
        #             result = self.unsharp_mask(result)
        #         else:
        #             result = cv2.GaussianBlur(result, (3,  3), 1)


        cv2.imshow("match result", result)

        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints in the image

        detector = cv2.ORB_create()

        # extract features from the image using SIFT approach
        (kps, features) = detector.detectAndCompute(gray, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays for ease of use
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:  # .75
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)  # 4.0

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def stitch_two_images_without_restriction(self, imageA, imageB, ratio=0.75, reprojThresh=4.0):

        bordersize = imageA.shape[1]

        # CAMBIAR LOS VALORES DENTRO DE VALUE
        imageB = cv2.copyMakeBorder(imageB, top=0, bottom=0, left=bordersize, right=bordersize,
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])


        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ok")

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # If M is None then it means that there is no overlap between two images
        if M is None:
            return None
        # Otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M

        result = cv2.warpPerspective(imageA, H,
                                     (int((imageA.shape[1] + imageB.shape[1])*1.5), int((imageA.shape[0]+imageB.shape[0])*1.5)), cv2.RANSAC)

        # Add imgB to the stitched plane
        for x, y in np.ndindex((imageB.shape[0], imageB.shape[1])):
            if sum(result[x, y, :]) == 0:
                result[x, y] = imageB[x, y]

        # Remove the black border of the final image
        result = self.crop(result)

        print("cropped ") # , result.shape[0], result.shape[1])

        return result

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c
        # https://mc.ai/panorama-formation-using-image-stitching-using-opencv/

    def stitch_images(self, imageA, imageB, ratio=0.75, reprojThresh=4.0):

        bordersize = imageA.shape[1]

        # CAMBIAR LOS VALORES DENTRO DE VALUE
        imageB = cv2.copyMakeBorder(imageB, top=0, bottom=0, left=bordersize, right=bordersize,
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])


        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ok")

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # If M is None then it means that there is no overlap between two images
        if M is None:
            return None
        # Otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M

        h1, w1 = imageB.shape[:2]
        h2, w2 = imageA.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        # print("pts:", pts)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

        result = cv2.warpPerspective(imageA, Ht.dot(H), (xmax - xmin, ymax - ymin))

        resizedB = np.zeros((result.shape[0], result.shape[1], 3), np.uint8)

        resizedB[t[1]:t[1] + h1, t[0]:w1 + t[0]] = imageB
        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        k1 = (kernel == 1).astype('uint8')
        mask = cv2.erode(mask, k1, borderType=cv2.BORDER_CONSTANT)

        mask_inv = cv2.bitwise_not(mask)

        difference = cv2.bitwise_or(resizedB, resizedB, mask=mask_inv)

        result2 = cv2.bitwise_and(result, result, mask=mask)

        result = cv2.add(result2, difference)

        result = self.crop(result)

        return result