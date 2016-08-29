"""Road Detection
"""

import argparse
import cv2
import numpy as np
from scipy.linalg import block_diag

class LaneDetector:
    def __init__(self, road_horizon, prob_hough=True):
        self.prob_hough = prob_hough
        self.vote = 50
        self.roi_theta = 0.3
        self.road_horizon = road_horizon

    def _standard_hough(self, img, init_vote):
        # Hough transform wrapper to return a list of points like PHough does
        lines = cv2.HoughLines(img, 1, np.pi/180, init_vote)
        points = [[]]
        for l in lines:
            for rho, theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)
                points[0].append((x1, y1, x2, y2))
        return points

    def _base_distance(self, x1, y1, x2, y2, width):
        # compute the point where the give line crosses the base of the frame
        # return distance of that point from center of the frame
        if x2 == x1:
            return (width*0.5) - x1
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        base_cross = -c/m
        return (width*0.5) - base_cross

    def _scale_line(self, x1, y1, x2, y2, frame_height):
        # scale the farthest point of the segment to be on the drawing horizon
        if x1 == x2:
            if y1 < y2:
                y1 = self.road_horizon
                y2 = frame_height
                return x1, y1, x2, y2
            else:
                y2 = self.road_horizon
                y1 = frame_height
                return x1, y1, x2, y2
        if y1 < y2:
            m = (y1-y2)/(x1-x2)
            x1 = ((self.road_horizon-y1)/m) + x1
            y1 = self.road_horizon
            x2 = ((frame_height-y2)/m) + x2
            y2 = frame_height
        else:
            m = (y2-y1)/(x2-x1)
            x2 = ((self.road_horizon-y2)/m) + x2
            y2 = self.road_horizon
            x1 = ((frame_height-y1)/m) + x1
            y1 = frame_height
        return x1, y1, x2, y2

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roiy_end = frame.shape[0]
        roix_end = frame.shape[1]
        roi = img[self.road_horizon:roiy_end, 0:roix_end]
        blur = cv2.medianBlur(roi, 5)
        contours = cv2.Canny(blur, 60, 120)

        if self.prob_hough:
            lines = cv2.HoughLinesP(contours, 1, np.pi/180, self.vote, minLineLength=30, maxLineGap=100)
        else:
            lines = self.standard_hough(contours, self.vote)

        if lines is not None:
            # find nearest lines to center
            lines = lines+np.array([0, self.road_horizon, 0, self.road_horizon]).reshape((1, 1, 4))  # scale points from ROI coordinates to full frame coordinates
            left_bound = None
            right_bound = None
            for l in lines:
                # find the rightmost line of the left half of the frame and the leftmost line of the right half
                for x1, y1, x2, y2 in l:
                    theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
                    if theta > self.roi_theta:  # ignore lines with a small angle WRT horizon
                        dist = self._base_distance(x1, y1, x2, y2, frame.shape[1])
                        if left_bound is None and dist < 0:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is None and dist > 0:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
                        elif left_bound is not None and 0 > dist > left_dist:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is not None and 0 < dist < right_dist:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
            if left_bound is not None:
                left_bound = self._scale_line(left_bound[0], left_bound[1], left_bound[2], left_bound[3], frame.shape[0])
            if right_bound is not None:
                right_bound = self._scale_line(right_bound[0], right_bound[1], right_bound[2], right_bound[3], frame.shape[0])

            return [left_bound, right_bound]



class LaneTracker:
    def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale, process_cov_parallel=0, proc_noise_type='white'):
        self.n_lanes = n_lanes
        self.meas_size = 4 * self.n_lanes
        self.state_size = self.meas_size * 2
        self.contr_size = 0

        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, self.contr_size)
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((self.meas_size, self.state_size), np.float32)
        for i in range(self.meas_size):
            self.kf.measurementMatrix[i, i*2] = 1

        if proc_noise_type == 'white':
            block = np.matrix([[0.25, 0.5],
                               [0.5, 1.]], dtype=np.float32)
            self.kf.processNoiseCov = block_diag(*([block] * self.meas_size)) * proc_noise_scale
        if proc_noise_type == 'identity':
            self.kf.processNoiseCov = np.eye(self.state_size, dtype=np.float32) * proc_noise_scale
        for i in range(0, self.meas_size, 2):
            for j in range(1, self.n_lanes):
                self.kf.processNoiseCov[i, i+(j*8)] = process_cov_parallel
                self.kf.processNoiseCov[i+(j*8), i] = process_cov_parallel

        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32) * meas_noise_scale

        self.kf.errorCovPre = np.eye(self.state_size)

        self.meas = np.zeros((self.meas_size, 1), np.float32)
        self.state = np.zeros((self.state_size, 1), np.float32)

        self.first_detected = False

    def _update_dt(self, dt):
        for i in range(0, self.state_size, 2):
            self.kf.transitionMatrix[i, i+1] = dt

    def _first_detect(self, lanes):
        for l, i in zip(lanes, range(0, self.state_size, 8)):
            self.state[i:i+8:2, 0] = l
        self.kf.statePost = self.state
        self.first_detected = True

    def update(self, lanes):
        if self.first_detected:
            for l, i in zip(lanes, range(0, self.meas_size, 4)):
                if l is not None:
                    self.meas[i:i+4, 0] = l
            self.kf.correct(self.meas)
        else:
            if lanes.count(None) == 0:
                self._first_detect(lanes)

    def predict(self, dt):
        if self.first_detected:
            self._update_dt(dt)
            state = self.kf.predict()
            lanes = []
            for i in range(0, len(state), 8):
                lanes.append((state[i], state[i+2], state[i+4], state[i+6]))
            return lanes
        else:
            return None


class LaneFinder:
    """
    implement using algorithm from journal
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

    def lane_detection(self, img):
        pass

    def curvature_estimation(self, img):
        pass

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

    ticks = 0

    lt = LaneTracker(2, 0.1, 500)
    ld = LaneDetector(180)
    while cap.isOpened():
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()

        ret, frame = cap.read()

        predicted = lt.predict(dt)

        lanes = ld.detect(frame)

        if predicted is not None:
            cv2.line(frame, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 0, 255), 5)
            cv2.line(frame, (predicted[1][0], predicted[1][1]), (predicted[1][2], predicted[1][3]), (0, 0, 255), 5)

        lt.update(lanes)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

