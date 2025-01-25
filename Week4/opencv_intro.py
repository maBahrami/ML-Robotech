import cv2

print(cv2.__version__)


img = cv2.imread(r"Week4\Datasets\Atlas.webp")
#print(img)

print(img.shape)
print(img.dtype)

roi = img[150:350, 200:1000]
cv2.imshow("roi image", roi)
cv2.waitKey()

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB image", rgb_img)
cv2.waitKey()




 
