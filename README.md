# LacmusTflite
The article considers a simplified solution to the problem of detecting small objects occupying less than 1% of the image area. EfficientNet and MobileNet from Tensorflow library pre-trained on data imagenet are used as feature extraction algorithms. The feature map is taken from the last layers of the neural network, class prediction is done along the feature vector in depth. Due to the small size objects, coordinates are not refined. Thus, it simplifies network architecture and reduces the number of calculations. This feature is especially important for mobile applications, because it allows to reduce detection time and battery consumption.

Experiments with tflite tiny object detection.

Data is here: https://cloud.mail.ru/public/2k53/2bJVwYSa7/
