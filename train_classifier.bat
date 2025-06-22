cd /d "C:\Users\user\Downloads\haar_training_project"
"C:\Users\user\Downloads\opencv\build\x64\vc14\bin\opencv_traincascade.exe" ^
-data classifier_output ^
-vec samples.vec ^
-bg neg.txt ^
-numPos 196 ^
-numNeg 113 ^
-numStages 10 ^
-w 100 -h 100 ^
-featureType HAAR
pause