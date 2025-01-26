import cv2
import numpy as np

img = np.ones([512, 512, 3])


# Colors (B, G, R)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

centerB = (366, 306) # x,y
centerG = (140, 306) # x,y
centerR = (256, 106) # x,y

axesOutter = (100, 100)
axesInner = (38, 38)
angle = 0. # clockwise, first axis, starts horizontal

# ------- blue ----------
cv2.ellipse(img, centerB, axesOutter, angle, -60, 240, (255,0,0),-25)
cv2.ellipse(img, centerB, axesInner, angle, -60, 240, WHITE,-25)

# ------- green -----------
cv2.ellipse(img, centerG, axesOutter, angle, 0., 300, (0,255,0),-25)
cv2.ellipse(img, centerG, axesInner, angle, 0., 300, WHITE,-25)

# ------- red ------------
cv2.ellipse(img, centerR, axesOutter, angle, 120, 420, (0,0,255),-25)
cv2.ellipse(img, centerR, axesInner, angle, 120, 420, WHITE,-25)


cv2.imshow("openCV logo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img)