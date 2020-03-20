import cv2
import numpy as np

def matcher(self, orb):
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(self.descriptor, orb.descriptor, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * .15)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(self.img, self.keypoints, orb.img, orb.keypoints, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = self.keypoints[match.queryIdx].pt
        points2[i, :] = orb.keypoints[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = orb.img.shape
    im1Reg = cv2.warpPerspective(self.img, h, (width, height))
    print(h)

    return im1Reg

def matcher2(self, orb):
    # img = cv2.drawKeypoints(self.img, self.keypoints, np.array([]), (0, 0, 255),
    #                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("keypoints", img)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(self.descriptor, orb.descriptor)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # img2 = cv2.drawMatches(self.img, self.keypoints, orb.img, orb.keypoints, matches, None, flags=2)

    width = self.img.shape[1] + orb.img.shape[1]
    height = self.img.shape[0] + orb.imgimg.shape[0]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = self.keypoints[match.queryIdx].pt
        points2[i, :] = orb.keypoints[match.trainIdx].pt

    h, status = cv2.findHomography(points1, points2, cv2.RANSAC)

    result = cv2.warpPerspective(orb.img, h, (width, height))
    result[0:self.img.shape[0], 0:self.img.shape[1]] = self.img
    # https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

    return result


def mix_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if (np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and
                            np.array_equal(warpedImage[j, i], np.array([0, 0, 0]))):
                        # print "BLACK"
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if (np.array_equal(warpedImage[j, i], [0, 0, 0])):
                            # print "PIXEL"
                            warpedImage[j, i] = leftImage[j, i]
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                                bl, gl, rl = leftImage[j, i]
                                warpedImage[j, i] = [bl, gl, rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage



class ORB():

    def __init__(self, img, orig):
        self.img = img
        self.org = orig
        self.descriptor = None
        self.keypoints = self.orb(img)
        # self.shi_tomasi = self.shi_tomasi(img)

    def shi_tomasi(self, img):
        corners_img = cv2.goodFeaturesToTrack(img, 1000, 0.01, 10)

        corners_img = np.int0(corners_img)

        for corners in corners_img:
            x, y = corners.ravel()
            cv2.circle(img, (x, y), 3, [0, 255, 0], -1)

        # self.surf()

    def crop(self, img):
        return

    def matcher(self, orb):
        bf = cv2.BFMatcher()

        match = bf.knnMatch(self.descriptor, orb.descriptor, 2)
        matches = []

        for m in match:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * .75:  # .75
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * .15)
        matches = matches[:numGoodMatches]

        # img2 = cv2.drawMatches(self.img, self.keypoints, orb.img, orb.keypoints, matches, None, flags=2)

        width = self.img.shape[1] + orb.img.shape[1]
        height = self.img.shape[0] + orb.img.shape[0]

        # points1 = np.zeros((len(matches), 2), dtype=np.float32)
        # points2 = np.zeros((len(matches), 2), dtype=np.float32)

        # for i, match in enumerate(matches):
        # points1[i, :] = self.keypoints[match.queryIdx].pt
        # points2[i, :] = orb.keypoints[match.trainIdx].pt
        points1 = np.float32([self.keypoints[i].pt for (_, i) in matches])
        points2 = np.float32([orb.keypoints[i].pt for (i, _) in matches])

        h, status = cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)

        result = cv2.warpPerspective(self.img, h, (width, height))
        # result[0:self.img.shape[0], 0:self.img.shape[1]] = self.img
        # https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

        for x, y in np.ndindex((orb.img.shape[0], orb.img.shape[1])):
            if result[x, y] == 0:
                result[x, y] = orb.img[x, y]

        # Remove the black border of the final image
        # result = Commons.crop_image(result, 0)

        return result
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c
        # https://mc.ai/panorama-formation-using-image-stitching-using-opencv/

    def orb(self, img):
        orb = cv2.ORB_create()

        keypoints, descriptor = orb.detectAndCompute(img, None)

        self.descriptor = descriptor

        return keypoints