import cv2

#print(cv2.__version__)


#img = cv2.imread(r"Week4\Datasets\Atlas.webp")
#print(img)

"""
print(img.shape)
print(img.dtype)

roi = img[150:350, 200:1000]
cv2.imshow("roi image", roi)
cv2.waitKey()

img2 = cv2.imread(r"Week4\Datasets\palette.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB image", rgb_img2)
cv2.waitKey()
"""

""""
img2 = cv2.imread(r"Week4\Datasets\palette.jpg")
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow("RGB image", gray_img2)
print(gray_img2.shape)
cv2.waitKey()

blue_channel = img2[:, :, 0]
cv2.imshow("nlue channel", blue_channel)
cv2.waitKey()

cv2.destroyAllWindows()


cv2.imwrite("palete_blueChannel.jpg", blue_channel)
"""


cap = cv2.VideoCapture(r"C:\Users\mabah\Desktop\ML_wee4\Artemis.mp4")

while True:
    ret, frame = cap.read()

    if frame is None: break

    cv2.imshow("video", frame)
    
    if cv2.waitKey(30) == ord("q"): break









