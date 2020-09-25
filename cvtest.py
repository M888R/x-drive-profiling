import cv2
import numpy as np

m_to_px = 20
img = np.zeros((500, 500, 3), np.uint8)

pos = np.array([10.0, 10.0])
size = np.array([50, 50])

while True:
    tmp = img.copy()

    scaled_pos = pos * m_to_px
    x, y = scaled_pos
    left_corner = np.array([250, 250]) - size / 2 + scaled_pos
    bottom_right_corner = np.array([250, 250]) + size / 2 + scaled_pos

    cv2.putText(tmp, f'x: {x}, y: {y}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
    cv2.rectangle(tmp, tuple(left_corner.astype(int)), tuple(bottom_right_corner.astype(int)), (255, 0, 0))

    # pos += (np.array([1, 1]) * 0.01)

    cv2.imshow('imgg', tmp)
    cv2.waitKey(5)