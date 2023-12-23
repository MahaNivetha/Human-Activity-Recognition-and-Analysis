

### Human Activity Recognition and Analysis

## Overview

This dataset comprises activity recognition data from adult subjects aged between 70 to 95 years. The dataset is intended for analysis related to human activity recognition, and it includes various features such as sensor data and activity labels.

## Content

The dataset consists of the following files:

- `your_modified_file.csv`: The main dataset file in CSV format.

## Prerequisites

To run the analysis on this dataset, you will need:

- MATLAB installed on your system.

## Usage

1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/anshtanwar/adult-subjects-70-95-years-activity-recognition/data).
2. Extract the contents to your local machine.
3. Load the dataset in MATLAB:

    ```matlab
    % Load data
    data = readtable("/path/to/your_modified_file.csv");
    ```

4. Choose an analysis option and run the script:

    ```matlab
    run_activity_analysis_script
    ```
## Analysis 

### 1. User Interaction:
   - Users are prompted to choose an analysis option:
     - **Option 1:** Enter date manually
     - **Option 2:** Enter activity label
     - **Option 3:** Enter specific time

### 2. Data Filtering:
   - Depending on the user's choice, the script filters the dataset to focus on specific dates, activity labels, or time.

### 3. Activity Visualization:
   - A bar plot is generated to visualize the count of different activity labels on the selected date or time.

### 4. Dynamic Mode Decomposition (DMD):
   - DMD is applied to analyze the temporal dynamics of selected data columns (`back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`).
   - DMD results, including mode frequencies, reconstructed data, and mode amplitude evolution over time, are displayed.

### 5. Sensitivity Analysis:
   - The script performs a sensitivity analysis by perturbing the data and comparing the original and perturbed DMD eigenvalues.

### 6. Statistical Analysis:
   - Mean, standard deviation, and range statistics are calculated for sensor data (`back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`).

### 7. Correlation Analysis:
   - A correlation matrix heatmap is generated to visualize the relationships between different sensor data columns.

### 8. Predictive Modeling (Support Vector Machine - SVM):
   - The script uses a multi-class SVM to build a predictive model based on sensor data.
   - The SVM decision boundary, support vectors, accuracy, and confusion matrix are displayed.

### 9. Visualization:
   - Various plots, including DMD mode frequencies, mode amplitude evolution, and SVM decision boundary, are displayed for visual analysis.

## Note:
- The script provides a comprehensive analysis of activity recognition data, incorporating advanced techniques like DMD and SVM.
- It includes visualizations to aid in understanding temporal dynamics, sensitivity, statistical properties, and predictive modeling of the dataset.

## Instructions for Users:
- Users need to interact with the script by choosing an analysis option.
- Ensure that the data file path is correctly specified.
- The script assumes specific column names and formats in the dataset.

Users can utilize this script to gain insights into the temporal patterns, statistical properties, and predictive modeling aspects of the activity recognition dataset. Adjustments can be made based on specific dataset characteristics and analysis requirements.
