import cv2

img = cv2.imread(r"Week7\reference\captcha_sample.PNG")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(img.shape)
#print(gray.shape)

_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(cnts[0].shape)


#cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
