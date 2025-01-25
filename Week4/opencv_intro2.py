import cv2

img = cv2.imread(r"C:\Users\mabah\Desktop\ML_Robotech\Week4\Datasets\Atlas.webp")

cv2.line(img, (10, 400), (167, 234), (255, 0, 0), 3)
cv2.rectangle(img, (550, 100), (650, 200), (255, 0, 0), 3)
cv2.circle(img, (550, 500), 100, (255, 0, 0), 3)
cv2.putText(img, "Atlas Robot", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)


cv2.imshow("my image", img)
cv2.waitKey()
cv2.destroyAllWindows()
