# Comprehensive Binary Classification with Datasets Merging

This project involves building and evaluating machine learning models for a binary classification problem. The dataset used is split into two parts, `Training_part1.csv` and `Training_part2.csv`, which are merged to create a complete training dataset.

No description of the dataset fields is provided, but we only know that the "Class" column is the target variable. The "id" column is present in both tables and should be used for matching.

## Dataset Description

- `Training_part1.csv`: Contains the first part of the training data.
- `Training_part2.csv`: Contains the second part of the training data.
## Preprocessing

- Duplicate records in both parts of the training data are removed.
- Missing values are handled:
  - Numeric features (`int_features` and `float_features`) are filled with the mean of the respective columns.
  - Categorical features (`cat_features`) are filled with the mode of the respective columns.
- Categorical features are converted into dummy variables.

## Feature Selection

- Pearson correlation is used to identify highly correlated features.
- Features with a correlation greater than 0.95 are dropped to reduce multicollinearity.

## Models Used

- Logistic Regression with Cross-Validation
- Random Forest Classifier with Cross-Validation
- Linear Support Vector Machine (SVM) with Cross-Validation

## Evaluation Metrics

- Accuracy: Measures the overall accuracy of the model predictions.
- Classification Reports: Precision, Recall, F1-score, Support, Macro Average, and Weighted Avg.


## File Structure

- `data/`: Directory containing the dataset files.
  - `Training_part1.csv`: First part of the training data.
  - `Training_part2.csv`: Second part of the training data.
- `evaluation/`: Directory containing scripts for evaluating model performance.
  - `evaluation_metrics.py`: Contains functions for generating evaluation metrics.
- `models/`: Directory containing scripts for different classification models.
  - `logistic_regression.py`: Contains functions for logistic regression modeling.
  - `random_forest.py`: Contains functions for random forest modeling.
  - `svm_classifier.py`: Contains functions for support vector machine modeling.
- `preprocessing/`: Directory containing scripts for data cleaning and preprocessing.
  - `data_cleaning.py`: Contains functions for cleaning and merging the datasets.
  - `feature_selection.py`: Contains functions for selecting relevant features.
- `main.py`: Main Python script that orchestrates data preprocessing, model training, and evaluation.
- `Dockerfile`: Contains instructions for building the Docker image.
- `docker-compose.yml`: Defines services for running the application using Docker Compose.
- `requirements.txt`: Lists the required dependencies for the project.
- `README.md`: Documentation for the project.

## Running the Code

### Using Python

1. Install the required dependencies listed in `requirements.txt` using pip:
   ```bash
   pip install -r requirements.txt
2. Run the main.py script:
   ```bash
   python main.py

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t data-merging-classification .
2. Run the Docker container:
   ```bash
   docker run data-merging-classification
3. Alternatively, use Docker Compose:
   ```bash
   docker-compose up