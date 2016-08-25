"""Road Detection
"""

import argparse
import cv2
import numpy as np
from scipy import weave


class LaneFinderFitline:
    """
    simple implementation using fitline
    - without using ROI
    - need classifier to improve detection
    """
    def __init__(self):
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
        # img[0:int(ht * 6 / 10), :] = (0, 0, 0)

        # threshold, remain the white or yellow line
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 30, 255,
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

class LaneFinder:
    """
    implement using algorith from journal
    New Lane Detection Algorithm for Autonomous Vehicles Using computer vision
    Quoc-Bao Truong and Byung-Ryong Lee
    """
    def __init__(self):
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
        # img[0:int(ht * 6 / 10), :] = (0, 0, 0)

        # threshold, remain the white or yellow line
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # gaussian filter for reduce noise
        gaus = cv2.GaussianBlur(gray, (7, 7), 0)

        median = cv2.medianBlur(gaus, 5)

        # histogram equalization to improve contrast and brightness
        hist = cv2.equalizeHist(median)

        # _, bw2 = cv2.threshold(hist, 128, 255, cv2.THRESH_BINARY)

        # canny edge detection
        canny = cv2.Canny(hist, 50, 150)

        return canny

    def _thinningIteration(self, im, iter):
        """
        currently very slow if using thinning
        :param im:
        :param iter:
        :return:
        """

        I, M = im, np.zeros(im.shape, np.uint8)
        expr = """
    	for (int i = 1; i < NI[0]-1; i++) {
    		for (int j = 1; j < NI[1]-1; j++) {
    			int p2 = I2(i-1, j);
    			int p3 = I2(i-1, j+1);
    			int p4 = I2(i, j+1);
    			int p5 = I2(i+1, j+1);
    			int p6 = I2(i+1, j);
    			int p7 = I2(i+1, j-1);
    			int p8 = I2(i, j-1);
    			int p9 = I2(i-1, j-1);
    			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
    			         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
    			         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
    			         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
    			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
    			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
    			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
    			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
    				M2(i,j) = 1;
    			}
    		}
    	}
    	"""

        weave.inline(expr, ["I", "iter", "M"])
        return I & ~M

    def thinning(self, img):
        # Zhang-Suen algorithm. for thinning image
        dst = img.copy() / 255
        prev = np.zeros(img.shape[:2], np.uint8)
        diff = None

        while True:
            dst = self._thinningIteration(dst, 0)
            dst = self._thinningIteration(dst, 1)
            diff = np.absolute(dst - prev)
            prev = dst.copy()
            if np.sum(diff) == 0:
                break

        return dst * 255


if __name__ == '__main__':
    """ Example implementation
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to the video", required=True)

    args = vars(ap.parse_args())

    # if args["video"] is None and args["image"] is None:
    #     print("invalid parameter")
    #     exit(0)

    cap = cv2.VideoCapture(args["video"])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        ld = LaneFinder()
        img = ld.prepare(frame)
        img = ld.thinning(img)

        # ld = LaneFinderFitline()
        # img = ld.detect_lane(frame)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

