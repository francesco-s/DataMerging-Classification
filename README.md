# Comprehensive Binary Classification with datasets merging

This project involves building and evaluating machine learning models for a binary classification problem. 
The dataset used is split into two parts, `Training_part1.csv` and `Training_part2.csv`, which are merged to create a full training dataset.

No description of the dataset fields is provided, but we only know the label: “class” column is the target variable.
The “id” column is present in both tables and should be used for matching.

## Dataset Description

- `Training_part1.csv`: Contains the first part of the training data.
- `Training_part2.csv`: Contains the second part of the training data.
- `Training_full.csv`: Merged dataset combining both parts of the training data.

## Preprocessing

- Duplicate records in both parts of the training data are removed.
- Missing values are handled:
  - Numeric features (`int_feature` and `float_feature`) are filled with the mean of the respective columns.
  - Categorical features (`cat_features`) are filled with the mode of the respective columns.
- Categorical features are converted into dummy variables.

## Feature Selection

- Pearson correlation is used to identify highly correlated features.
- Features with correlation greater than 0.95 are dropped.

## Models Used

- Logistic Regression
- Random Forest Classifier
- Linear Support Vector Machine (SVM)

## Evaluation Metrics

- Accuracy: Measures the overall accuracy of the model predictions.
- F1-Score: Measures the balance between precision and recall for each class.

## File Structure

- `data/`: Directory containing the dataset files.
- `main.py`: Main Python script containing the code for preprocessing, model training, and evaluation.

## Running the Code

1. Install the required dependencies using pip:
  ```
  pip install pandas scikit-learn matplotlib
  ```
2. Run the `main.py` script:
  ```
  python main.py
  ```
## Results

- Model accuracies, classification reports, confusion matrices, and F1 scores are printed to the console.
- F1 scores for each model and class are visualized using a bar plot.
