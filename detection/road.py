"""Road Detection

simple implementation using fitline
- without using ROI
- need classifier to improve detection
"""

import argparse
import cv2
import numpy as np


class LineFinder:

    def __init__(self):
        self.img = None
        # accumulator resolution parameters
        self.deltaRho = 1
        self.deltaTheta = np.pi / 180.0

        # filter
        self.invtheta = 180 / np.pi
        self.angle = 20

    def set_acc_resolution(self, d_rho, d_theta):
        self.deltaRho = d_rho
        self.deltaTheta = d_theta


    def prepare(self, img):
        """ prepare """
        ht, wd, dp = img.shape

        # only care about the horizont block, filter out up high block
        img[0:int(ht / 2), :] = (0, 0, 0)

        # threshold, remain the white or yellow line
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 128, 255,
                                    cv2.THRESH_BINARY)
        return thresh

    def detect(self, thresh):
        """ use contours and fitline to detect line """
        contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None:
            return []
        return [cv2.fitLine(cnt,cv2.cv.CV_DIST_L2, 0, 0.01, 0.01) for cnt in contours]

    def find_distance(self, img, line):
        """ find from pp (x,y) and to pp (x,y) """
        ht, wd, dp = img.shape
        [vx, vy, x, y] = line

        # y = ax + b
        x1, y1 = int(x-vx*int(ht/2)), int(y-vy*int(ht/2))
        x2, y2 = int(x+vx*int(ht/2)), int(y+vy*int(ht/2))

        # boundary check
        x1 = x1 if x1 <= wd else x1 if x1 >= 0 else 0
        x2 = x2 if x2 <= wd else x2 if x2 >= 0 else 0
        y1 = y1 if y1 <= ht else y1 if y1 >= int(ht/2) else int(ht/2)
        y2 = y2 if y2 <= ht else y2 if y2 >= int(ht/2) else int(ht/2)
        return x1, y1, x2, y2

    def find_angle(self, img, line ):
        """ angle """
        x1, y1, x2, y2 = self.find_distance(img, line)
        dx, dy = x2 - x1, y2 - y1
        angle = np.arctan2(dy, dx) * self.invtheta
        return angle, x1, y1, x2, y2

    def draw(self, img, lines):
        """ draw results """
        for line in lines:
            angle, x1, y1, x2, y2 = self.find_angle(img, line)
            if angle <= self.angle and angle >= - self.angle:
                return img
            cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2) # draw red colour
        return img

    def show(self, img):
        """ show img """
        cv2.imshow('fitline', img)
        cv2.waitKey(1)

    def detect_lane(self, img):
        """
        detect lane
        :param binary:
        :return: image
        """

        thresh = self.prepare(img.copy())
        lines = self.detect(thresh)

        img = self.draw(img, lines)
        # self.show(img)

        return img



if __name__ == '__main__':
    """ Example implementation
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to the video", required=True)
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["video"])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        ld = LineFinder()
        img = ld.detect_lane(frame)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

