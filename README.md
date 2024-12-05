# Weapons and Knives Detector with YOLOv8  

This project demonstrates a comprehensive solution for detecting weapons, such as knives and guns, in images, videos, and live webcam feeds using the YOLOv8 object detection model. The system provides real-time notifications and actionable guidance through an integrated web application.  

## Features  
- **Object Detection**: Utilizes YOLOv8 for real-time identification of weapons.  
- **Web Application**: Built with Flask, providing an intuitive interface for uploading media and viewing results.  
- **Real-Time Notifications**: Sends alerts to users via Telegram when a weapon is detected.  
- **AI-Powered Guidance**: Integrates CharacterAI to offer contextual suggestions upon detection.  
- **Versatility**: Supports static images, video files, and live webcam feeds.  

## Technologies Used  
- **YOLOv8**: A state-of-the-art object detection model.  
- **Flask**: Lightweight web framework for building the application interface.  
- **Telegram API**: For sending instant notifications.  
- **CharacterAI**: Provides AI-driven recommendations to users.  
- **OpenCV**: For image and video processing.  
- **TensorFlow & PyTorch**: Deep learning frameworks for training and fine-tuning detection models.  

## How It Works  
1. **Upload Media**: Users can upload images or videos via the web interface or enable live webcam monitoring.  
2. **Detection**: The YOLOv8 model identifies weapons and marks them with bounding boxes.  
3. **Notifications**: Alerts are sent through Telegram with detection results and actionable suggestions.  
4. **Real-Time Monitoring**: The system provides live feedback for webcam streams.  

## Acknowledgments  
This project is based on the **Weapons and Knives Detector with YOLOv8** model by [João Assalim](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8).  
I have utilized this model as a foundation and made modifications to adapt it for my use case, including integrating real-time notifications, a web interface, and AI-driven recommendations.
