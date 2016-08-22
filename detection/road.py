"""Road Detection

Processing road detection
- set the ROI for the image
- Canny algorithm
- Process hough transform
   Hough tranform for line detection with feedback
   Increase by 25 for the next frame if we found some lines.
   This is so we don't miss other lines that may crop up in the next frame
   but at the same time we don't want to start the feed back loop from scratch.
- Draw the lines
- Create LineFinder instance
- Set probabilistic Hough parameters
- Detect lines
- bitwise AND of the two hough images
- Set probabilistic Hough parameters
- Writer the frame into the file
"""

import argparse
import cv2
import numpy as np


class LineFinder:
    PI = 3.1415926

    def __init__(self):
        self.img = None
        self.lines = []
        # accumulator resolution parameters
        self.deltaRho = 1
        self.deltaTheta = LineFinder.PI / 180.0

        # minimum number of votes that a line
        # must receive before being considered
        self.minVote = 10

        # min length for a line
        self.minLength = 0.0

        # max allowed gap along the line
        self.maxGap = 0.0;

        # distance to shift the drawn lines down when using a ROI
        self.shift = 0

    def set_acc_resolution(self, d_rho, d_theta):
        self.deltaRho = d_rho
        self.deltaTheta = d_theta

    def set_min_vote(self, min_vote):
        self.minVote = min_vote

    def set_line_and_gap(self, length, gap):
        self.minLength = length
        self.maxGap = gap

    def set_shift(self, img_shift):
        self.shift = img_shift

    def find_lines(self, binary):
        self.lines = cv2.HoughLinesP(binary, self.deltaRho, self.deltaTheta, self.minVote, self.minLength, self.maxGap);
        return self.lines

    def draw_detected_lines(self, image, color):
        pass

    def remove_inconsistent_lines(self, orientations, percentage, delta):
        pass


if __name__ == '__main__':
    """ Example implementation
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to the video", required=True)
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["video"])

    houghVote = 200

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        rows, cols, ch = frame.shape
        # convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # set the ROI for the image
        # 0,image.cols/3,image.cols-1,image.rows - image.cols/3)
        roi = gray[rows/2:rows, 0:cols]

        contours = cv2.Canny(roi, 50, 250)
        contoursInv, thresContours = cv2.threshold(contours, 128, 255, cv2.THRESH_BINARY_INV)
        lines = []
        if houghVote < 1:  # we lost all lines.reset
            houghVote = 200
        else:
            houghVote += 25
        while len(lines) < 5 and houghVote > 0:
            lines = cv2.HoughLines(contours, 1, LineFinder.PI / 180, houghVote);
            houghVote -= 5

        result = roi.copy()
        roiRows, roiCols = roi.shape
        hough = roi.copy()

        for rho, theta in lines[0]:
            if (theta > 0.09 and theta < 1.48) or (theta < 3.14 and theta > 1.66):
                # filter to remove vertical and horizontal lines

                #point of intersection of the line with first row
                x1 = int(rho / np.cos(theta))
                y1 = 0

                # point of intersection of the line with last row
                x2 = int((rho - roiRows * np.sin(theta)) / np.cos(theta))
                y2 = roiRows
                # draw a white line
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 8)
                cv2.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 8)

        ld = LineFinder()
        ld.set_line_and_gap(60, 10);
        ld.set_min_vote(4);

        # detect line
        li = ld.find_lines(contours)
        houghP = roi.copy()
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image():
    pass
