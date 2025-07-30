# Diabetes Prediction Project Configuration
# ========================================

# Data Configuration
DATA_FILE = "diabetes.csv"
RANDOM_STATE = 1
TEST_SIZE = 0.3

# Model Configuration
MODEL_TYPE = "DecisionTree"
MAX_DEPTH = None  # None for unlimited depth
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

# Feature Configuration
CATEGORICAL_FEATURES = [
    "Gender", "Polyuria", "Polydipsia", "sudden weight loss", 
    "weakness", "Polyphagia", "Genital thrush", "visual blurring",
    "Itching", "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity", "class"
]

NUMERICAL_FEATURES = ["Age"]

TARGET_COLUMN = "class"

# Visualization Configuration
FIGURE_SIZE = (13, 10)
COLOR_PALETTE = "viridis"
PLOT_DPI = 300

# Age Group Configuration
AGE_GROUP_INTERVALS = 15
AGE_GROUP_START = 20

# Output Configuration
MODEL_OUTPUT_PATH = "models/"
RESULTS_OUTPUT_PATH = "results/"
PLOTS_OUTPUT_PATH = "plots/"

# Evaluation Metrics
METRICS_TO_CALCULATE = [
    "accuracy",
    "precision", 
    "recall",
    "f1_score",
    "confusion_matrix",
    "classification_report"
]

# Data Quality Thresholds
MAX_MISSING_VALUES_PERCENT = 5.0
MIN_FEATURE_VARIANCE = 0.01
MAX_CORRELATION_THRESHOLD = 0.95
