# Bone Age Estimator Using Xception Model

This project aims to estimate bone age from hand X-ray images using the Xception model. The estimation of bone age is critical for diagnosing growth disorders and planning appropriate treatments in the medical field. 

## Project Overview

This repository contains the code and resources for training a deep learning model to estimate the bone age of patients based on X-ray images of hands. The project utilizes the Xception model, a powerful convolutional neural network architecture, to achieve high accuracy in bone age estimation.

## Dataset

The dataset used for this project is the RSNA Bone Age dataset, which consists of X-ray images of hands along with corresponding bone age labels.

## Preprocessing Techniques

The following preprocessing techniques were applied to the X-ray images:
1. Laplacian of Gaussian (LOG)
2. Gaussian Gradient
3. Sobel Edge Detection
4. Canny Edge Detector
5. Thresholding

## Model Architecture

The Xception model was used as the base model. The architecture was modified by adding custom layers for the regression task of estimating bone age.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- scikit-image


## Usage

1. Download the RSNA Bone Age dataset and place it in the appropriate directory.
2. Update the paths in the code to point to the dataset location.
3. Run the preprocessing and model training script:
    ```bash
    python bone_age_estimator.py
    ```
4. Evaluate the model and visualize the results:
    ```bash
    python evaluate_model.py
    ```

## Code Structure

- `bone_age_estimator.py`: Main script for preprocessing, training, and evaluating the model.
- `evaluate_model.py`: Script for loading the trained model and evaluating its performance on the test set.
- `requirements.txt`: List of required Python packages.
- `logs/`: Directory for storing TensorBoard logs.
- `results.csv`: File containing the predictions on the test set.

## Results

The model was trained and validated on the RSNA Bone Age dataset. The performance was evaluated using Mean Absolute Error (MAE) in months. Visualization of the results shows the predicted bone ages compared to the actual ages.


## Acknowledgements

- The RSNA for providing the bone age dataset.
- The creators of the Xception model for their work on deep learning architectures.


---

