# Haar-Cascade-YZU-Vehicle-Detection
Based on the Haar Cascade method, a campus vehicle detection model for Yuan Ze University was developed using OpenCV’s createsamples and traincascade tools. The system aims to track vehicles within the campus to enhance environmental safety, and its performance is evaluated using a ROC curve.


Make sure you have the following installed:
Python 3.x
OpenCV (recommend version 3.4.11)
NumPy
Matplotlib (for ROC curve visualization)

You can install dependencies via pip:
pip install opencv-python numpy matplotlib

Installation & Setup:
1. Clone the Repository
   git clone https://github.com/VanessaTu0912/Haar-Cascade-YZU-Vehicle-Detection.git
   cd Haar-Cascade-YZU-Vehicle-Detection
2. Prepare Training Data
   Store all positive samples (images with vehicles) in a pos folder.
   Store all negative samples (background images) in a neg folder.
   Generate pos.txt (positive samples annotations) and neg.txt (negative samples file paths).
   (You can generate it by the code I uplode)
3. Create .vec File
   Use opencv_createsamples.exe:
   opencv_createsamples -info pos.txt -vec samples.vec -num 300 -w 100 -h 100
4. Train the Classifier
   Use opencv_traincascade.exe with parameters: (or change the path to where you download the
   file in train_classifier.bat and save it, then tape it to train the classifier)
   opencv_traincascade -data classifier_output -vec samples.vec -bg neg.txt -numPos 280 -numNeg
   144 -numStages 10 -w 100 -h 100 -featureType HAAR -minHitRate 0.985 -maxFalseAlarmRate 0.6 -
   maxWeakCount 50
5. Once training is complete, use the trained model (classifier_output/cascade.xml) in a
   detection script:
   python test.py
   you can yose the test_images with test_xml file to do the detection
6.Evaluation (ROC Curve)
  To evaluate model performance:
  python evaluate_haar_roc.py
  This will: Compare prediction results with ground truth, plot ROC curve ,and compute AUC score

📂 Folder Structure

haar_tranning_project/
│
├── pos/                      # Positive training images
├── neg/                      # Negative images
├── annotations/              # .xml of positive labeled image
├── test_image/               # Images for evaluation
├── test_xml/                 # .xml of test labeled image
├── classifier_output/        # Output trained model (cascade.xml)
├── test.py                   # Detection script
├── evaluate_haar_roc.py      # ROC evaluation
├── pos.txt  
├── neg.txt 
├── samples.vec               # Packed training data
└── README.md
