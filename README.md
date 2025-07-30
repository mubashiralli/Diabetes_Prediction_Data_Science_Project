# Diabetes Prediction Data Science Project

## 📋 Project Overview

This project implements a machine learning solution for predicting diabetes based on various health indicators and symptoms. Using a dataset of 520 patient records with 16 different health features, the project employs decision tree classification to predict whether a patient has diabetes or not.

## 🎯 Objective

The main objective is to develop a predictive model that can accurately classify patients as diabetic or non-diabetic based on their symptoms and health indicators, potentially assisting in early diabetes detection and healthcare decision-making.

## 📊 Dataset Description

**File:** `diabetes.csv`
- **Size:** 520 records (521 lines including header)
- **Features:** 16 health indicators + 1 target variable
- **Target Variable:** `class` (Positive/Negative for diabetes)

### Features:
1. **Age** - Patient's age
2. **Gender** - Male/Female
3. **Polyuria** - Excessive urination (Yes/No)
4. **Polydipsia** - Excessive thirst (Yes/No)
5. **Sudden weight loss** - (Yes/No)
6. **Weakness** - (Yes/No)
7. **Polyphagia** - Excessive hunger (Yes/No)
8. **Genital thrush** - (Yes/No)
9. **Visual blurring** - (Yes/No)
10. **Itching** - (Yes/No)
11. **Irritability** - (Yes/No)
12. **Delayed healing** - (Yes/No)
13. **Partial paresis** - Partial paralysis (Yes/No)
14. **Muscle stiffness** - (Yes/No)
15. **Alopecia** - Hair loss (Yes/No)
16. **Obesity** - (Yes/No)

## 🔧 Technologies Used

### Core Libraries:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms

### Additional Libraries:
- **pandas-profiling/ydata-profiling** - Automated exploratory data analysis
- **missingno** - Missing data visualization
- **Feature selection tools** - SelectKBest, chi2

## 🚀 Project Structure

```
Diabetes_Prediction_Data_Science_Project/
│
├── diabetes.csv          # Dataset containing patient health records
├── project.ipynb         # Main Jupyter notebook with analysis and model
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation (this file)
```

## 🔍 Analysis Workflow

### 1. Data Loading and Exploration
- Load the diabetes dataset using pandas
- Examine data structure, types, and basic statistics
- Check for missing values and duplicates

### 2. Data Preprocessing
- **Data Cleaning:** Remove duplicate records to reduce noise
- **Feature Engineering:** Convert categorical variables to numerical format
  - Yes/No → 1/0
  - Positive/Negative → 1/0
  - Male/Female → 0/1

### 3. Exploratory Data Analysis (EDA)
- **Correlation Analysis:** Generate heatmap to understand feature relationships
- **Gender Distribution:** Analyze diabetes prevalence by gender
- **Age Analysis:** 
  - Group ages into categories (G6-20, G21-35, etc.)
  - Visualize age distribution using histograms and box plots
- **Statistical Visualization:** Create various plots to understand data patterns

### 4. Machine Learning Model
- **Algorithm:** Decision Tree Classifier
- **Train-Test Split:** 70% training, 30% testing (random_state=1)
- **Model Training:** Fit the decision tree on training data
- **Prediction:** Generate predictions on test set
- **Evaluation:** Calculate accuracy metrics

## 📈 Key Features of the Analysis

1. **Comprehensive Data Preprocessing**
   - Handles categorical to numerical conversion
   - Removes data duplicates
   - Ensures data quality

2. **Visual Analytics**
   - Correlation heatmaps
   - Gender-based diabetes distribution
   - Age group analysis
   - Statistical distributions

3. **Machine Learning Pipeline**
   - Feature selection capabilities
   - Decision tree classification
   - Model evaluation metrics

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation Steps

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd Diabetes_Prediction_Data_Science_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook project.ipynb
   ```

## 📝 Usage

1. **Open the Jupyter notebook** (`project.ipynb`)
2. **Run cells sequentially** to:
   - Load and explore the data
   - Perform data preprocessing
   - Generate visualizations
   - Train the machine learning model
   - Evaluate model performance

3. **Customize the analysis** by:
   - Modifying visualization parameters
   - Trying different machine learning algorithms
   - Adjusting train-test split ratios
   - Adding new features or preprocessing steps

## 📊 Expected Results

The project provides:
- **Data Insights:** Understanding of feature correlations and distributions
- **Visual Analytics:** Clear visualizations of diabetes patterns
- **Predictive Model:** Decision tree classifier with accuracy metrics
- **Feature Importance:** Understanding of which symptoms are most predictive

## 🔮 Future Enhancements

1. **Model Improvements:**
   - Try other algorithms (Random Forest, SVM, Neural Networks)
   - Implement cross-validation
   - Hyperparameter tuning

2. **Advanced Analytics:**
   - Feature importance analysis
   - ROC curves and AUC metrics
   - Confusion matrix analysis

3. **Data Enhancements:**
   - Collect more diverse data
   - Handle class imbalance if present
   - Feature engineering for better predictions

## 📚 Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages include:
- pandas==2.0.3
- numpy==1.24.4
- matplotlib==3.7.5
- seaborn==0.12.2
- scikit-learn==1.3.2
- ydata-profiling==4.7.0

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new machine learning models
- Improving data visualization
- Enhancing documentation
- Optimizing code performance

## 📄 License

This project is open-source and available for educational and research purposes.

## 👨‍💻 Author

**Mubashir Ali**
- GitHub: [@mubashiralli](https://github.com/mubashiralli)

---

*This project demonstrates the application of data science techniques for healthcare prediction, showcasing the potential of machine learning in medical diagnosis assistance.*
