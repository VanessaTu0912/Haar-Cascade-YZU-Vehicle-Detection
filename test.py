import cv2
import os

# 讀取分類器
cascade_path = "classifier_output/cascade.xml"
if not os.path.exists(cascade_path):
    print(f"模型檔案不存在：{cascade_path}")
    exit()
classifier = cv2.CascadeClassifier(cascade_path)

# 讀取圖片
image_path = "test_image/LINE_ALBUM_test_250619_4.jpg"
img = cv2.imread(image_path)
if img is None:
    print(f"無法讀取圖片：{image_path}")
    exit()

# 若圖片過大，縮小一半
height, width = img.shape[:2]
if max(height, width) > 1500:
    img = cv2.resize(img, (width // 2, height // 2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 偵測物件
boxes = classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=25,
    minSize=(250, 250)
)
print(f"偵測到 {len(boxes)} 個物件")

# 繪製框框
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
