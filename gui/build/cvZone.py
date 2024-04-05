import math
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.interpolate import interp1d
import math



class LivePlot:
    """
    A class for real-time plotting in OpenCV.
    """

    def __init__(self, w=640, h=480, yLimit=[0, 100],
                 interval=0.001, invert=True):
        """
        Initialize the LivePlot object.

        :param w: Width of the plotting window
        :param h: Height of the plotting window
        :param yLimit: Y-axis limits [y_min, y_max]
        :param interval: Time interval for updating the plot
        :param invert: Whether to invert the y-axis
        :param char: A character to display on the plot for annotation
        """

        self.yLimit = yLimit
        self.w = w
        self.h = h
        self.invert = invert
        self.interval = interval
        self.imgPlot = np.zeros((self.h, self.w, 3), np.uint8)
        self.imgPlot[:] = 225, 225, 225
        self.xP = 0
        self.yP = 0
        self.yList = []
        self.xList = [x for x in range(0, 200)]
        self.ptime = 0
    
    def interpolate_points(self, x_points, y_points):
        if len(np.unique(x_points)) < 4 or len(y_points) < 4:
            return x_points, y_points  # Fallback to original points

        try:
            spline = interp1d(x_points, y_points, kind='cubic', fill_value="extrapolate")
            x_new = np.linspace(x_points[0], x_points[-1], num=len(x_points) * 10)
            y_new = spline(x_new)
            return x_new, y_new
        except ValueError as e:
            return x_points, y_points  # Fallback to original points

    def update(self, y, color=(0, 0, 255)):
        if time.time() - self.ptime > self.interval:
            self.imgPlot[:] = 225, 225, 225  # Clear the plot
            self.drawBackground()

            if self.invert:
                self.yP = int(np.interp(y, self.yLimit, [self.h, 0]))
            else:
                self.yP = int(np.interp(y, self.yLimit, [0, self.h]))

            self.yList.append(self.yP)
            if len(self.yList) > 200:
                self.yList.pop(0)

            self.xList = list(range(self.w - len(self.yList), self.w))

            # Interpolate for smoother curve
            x_new, y_new = self.interpolate_points(self.xList[-len(self.yList):], np.array(self.yList))
            bottom_left = [x_new[0], self.h if self.invert else 0]
            bottom_right = [x_new[-1], self.h if self.invert else 0]

            # Convert interpolated points for OpenCV polylines
            pts = np.array([x_new, y_new]).T.reshape(-1, 1, 2).astype(np.int32)
            bottom_left = np.array([bottom_left]).T.reshape(-1, 1, 2).astype(np.int32)
            bottom_right = np.array([bottom_right]).T.reshape(-1, 1, 2).astype(np.int32)
            pts = np.vstack([pts, bottom_right, bottom_left])
            

            # Fill the area under the curve
            cv2.fillPoly(self.imgPlot, [pts], color)


            # cv2.polylines(self.imgPlot, [pts], isClosed=False, color=color, thickness=2)


            self.ptime = time.time()

        return self.imgPlot

    def drawBackground(self):
        """
        Draw the static background elements of the plot.
        """

        cv2.rectangle(self.imgPlot, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)
        # cv2.line(self.imgPlot, (0, self.h // 2), (self.w, self.h // 2), (150, 150, 150), 2)

        # Draw grid lines and y-axis labels
        # for x in range(0, self.w, 50):
        #     cv2.line(self.imgPlot, (x, 0), (x, self.h), (50, 50, 50), 1)
        # for y in range(0, self.h, 50):
        #     cv2.line(self.imgPlot, (0, y), (self.w, y), (50, 50, 50), 1)
        #     y_label = int(self.yLimit[1] - ((y / 50) * ((self.yLimit[1] - self.yLimit[0]) / (self.h / 50))))
        #     cv2.putText(self.imgPlot, str(y_label), (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)

        # cv2.putText(self.imgPlot, self.char, (self.w - 100, self.h - 25), cv2.FONT_HERSHEY_PLAIN, 5, (150, 150, 150), 5)


if __name__ == "__main__":
    xPlot = LivePlot(w=1200, yLimit=[-100, 100], interval=0.01)
    x = 0
    while True:

        x += 1
        if x == 360: x = 0
        imgPlot = xPlot.update(int(math.sin(math.radians(x)) * 100))

        cv2.imshow("Image", imgPlot)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break