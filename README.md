# 😴 Drowsiness Detection Using YOLOv5 🚗

## Overview
This project utilizes the YOLOv5 architecture to develop an advanced drowsiness detection system. The model is designed to identify signs of driver drowsiness, such as closed eyes, yawning, and head movements, using a custom dataset. The goal is to enhance driver safety by providing timely alerts when drowsiness is detected.

## Performance Analysis

### Quantitative Results
The YOLOv5 model was trained and tested over 25 epochs, showing exceptional results:

- **Mean Average Precision (mAP)**: 0.97924 for the training dataset after 25 epochs.
- **Training Precision**: 0.95620
- **Training Recall**: 0.96873


## Model Performance Metrics
| Algorithm | Accuracy | Precision | Recall | F1 Score | Missing Detection Score | Inference Time (seconds) |
|-----------|----------|-----------|--------|----------|-------------------------|--------------------------|
| **YOLOv5**    | 0.90476  | 0.92857   | 0.92857| 0.92857  | 0.00000                 | 61.31154                 |


## Results
<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5/blob/main/results/PR_curve.png" alt="PR Curve" style="width: 400px;">
      <br>PR Curve
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5/blob/main/results/P_curve.png" alt="Precision Curve" style="width: 400px;">
      <br>Precision Curve
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5/blob/main/results/R_curve.png" alt="Recall Curve" style="width: 400px;">
      <br>Recall Curve
    </td>
  </tr>
  <tr>
     <td style="text-align: center;">
      <img src="https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5/blob/main/results/F1_curve.png" alt="F1 Curve" style="width: 400px;">
      <br>F1 Curve
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5/blob/main/results/confusion_matrix.png" alt="Confusion Matrix" style="width: 400px;">
      <br>Confusion Matrix
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5/blob/main/results/results.png" alt="Results" style="width: 400px;">
      <br>Results
    </td>
  </tr>
</table>

These outstanding results make YOLOv5 a promising model for real-time drowsiness detection.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ParthJohri/Drowsiness_Detection_Using_YOLOv5.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Drowsiness_Detection_Using_YOLOv5
   ```
3. Train your model on the custom dataset using colab notebook given, in the notebook directory.

4. Before running the model, make sure you have the yolov5 installed, after which run the real-time.py:
   ```sh
   python3 real_time.py
   ```
   
## Usage
1. Prepare your dataset with labeled images for training and testing.
2. Train the YOLOv5 model using the provided training script.
3. Test the model on the test dataset to evaluate its performance.
4. Deploy the model for real-time driver monitoring to detect signs of drowsiness.

## Future Work
- **Model Optimization**: Further refine the YOLOv5 model to enhance accuracy and reduce inference time.
- **Dataset Expansion**: Include more diverse datasets to improve model robustness and generalization.
- **Real-time Implementation**: Test the system in real-world driving scenarios to ensure practical applicability and reliability.

## Contributing
Contributions to improve the drowsiness detection system are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
