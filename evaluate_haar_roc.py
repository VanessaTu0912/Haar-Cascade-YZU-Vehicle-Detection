import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 模型路徑
model_path = r"C:\Users\user\Desktop\haar_training_project\classifier_output\cascade.xml"
classifier = cv2.CascadeClassifier(model_path)

# 測試資料夾
pos_dir = r"C:\Users\user\Desktop\haar_training_project\test_image\pos"
neg_dir = r"C:\Users\user\Desktop\haar_training_project\test_image\neg"

# 結果存放資料夾
output_dir = r"C:\Users\user\Desktop\haar_training_project\test_xml"
os.makedirs(output_dir, exist_ok=True)

# 儲存結果
y_true = []
y_score = []

def detect_and_score(image_path, label):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3
    )

    # 評分邏輯：偵測到（>0 box）算是正類的機率越高
    y_true.append(label)
    y_score.append(1 if len(boxes) > 0 else 0)

# 正樣本：應該要能偵測到
for filename in os.listdir(pos_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        detect_and_score(os.path.join(pos_dir, filename), 1)

# 負樣本：不應該偵測到
for filename in os.listdir(neg_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        detect_and_score(os.path.join(neg_dir, filename), 0)

# 畫 ROC 曲線
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='Haar ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # 對角線
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Haar Cascade')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
