# Aqueous-solubility-prediction


This project builds a machine learning pipeline to predict the aqueous solubility categories of organic molecules using their chemical and structural descriptors. By classifying molecules as **highly soluble, slightly soluble, or insoluble**, this tool aims to support early-stage molecule screening and accelerate materials and pharmaceutical R&D.

---

## Dataset Overview

The dataset for this project was provided as part of a university materials science course and curated for academic, non-commercial research purposes.

- **Dataset Size:** 1000+ organic molecules with measured aqueous solubility values (LogS).
- **Features:** Chemical and structural descriptors for each molecule, including molecular weight, number of hydrogen bond donors/acceptors, topological polar surface area, rotatable bonds, and other molecular fingerprints.
- **Classification Labels:** The continuous LogS values were converted into categorical labels for machine learning:
  - `Highly Soluble`: LogS >= 0
  - `Soluble`: -2 <= LogS < 0
  - `Slightly Soluble`: -4 <= LogS < -2
  - `Insoluble`: LogS < -4
- **Preprocessing Steps:**
  - Removed features with low variance or high correlation to reduce dimensionality while preserving predictive power.
  - Removed entries with missing values (NaNs) to ensure data integrity during model training.
  - Normalized continuous features for consistent scaling.
  - Analyzed feature correlations to inform model design and simplify inputs.
 
---

## Methods

The following machine learning methods were implemented and compared for predicting solubility categories:

- **K-Nearest Neighbors (KNN):** Utilized for its interpretability and effectiveness on small datasets.
- **Decision Tree:** Explored for its ability to handle non-linear feature interactions.
- **Support Vector Machine (SVM):** Tested for classification boundary performance.
- **Neural Network (TensorFlow Keras):** Built and trained a multi-layer perceptron for solubility classification, with normalization applied to input features during training.
- **Linear Regression:** Used in an exploratory phase to analyze linear relationships between molecular features and continuous solubility values.

**Visualization Tools:**
- Generated **heatmaps** to visualize feature correlations and understand data structure.
- Created **scatter plots** to compare predicted vs. actual solubility values during exploratory regression analysis.
- Used **Self-Organizing Maps (SOM)** for additional feature space visualization and pattern recognition.

**Evaluation Metrics:**
- Classification models were primarily evaluated using **accuracy**.
- Regression models used **R² (coefficient of determination)** and **Mean Squared Error (MSE)** for exploratory analysis.

These methods allowed for comprehensive experimentation and benchmarking to identify the most effective approach for **predicting aqueous solubility classes of organic molecules**.

---
## Results

After testing and comparing multiple models, the key findings of this project are:

- **Best Performing Models:**  
  The **K-Nearest Neighbors (KNN)** and **Decision Tree** classifiers achieved the highest test accuracy of approximately **65%** on the curated dataset, outperforming SVM and Neural Network models on this small and heterogeneous dataset.

- **Neural Network Insights:**  
  A basic multi-layer perceptron (MLP) neural network was built using TensorFlow Keras. However, it struggled to generalize on the dataset, achieving lower accuracy (~30%), highlighting the challenges of using deep learning models on small datasets without extensive tuning.

- **Exploratory Regression Analysis:**  
  Linear regression was performed to explore feature relationships with the continuous LogS values, with R² scores indicating moderate correlation between selected features and solubility, useful for understanding the dataset but not used for final predictions.

- **Visualization Findings:**  
  - **Heatmaps** revealed correlations among chemical descriptors, aiding feature understanding.
  - **Scatter plots** of predicted vs. actual values in regression provided insight into model limitations.
  - **SOM visualizations** were used to explore clustering patterns in the feature space.

Overall, this project demonstrated the feasibility of using machine learning to predict aqueous solubility categories of organic molecules and highlighted key considerations when applying AI to materials science challenges, including dataset limitations, feature importance, and model selection.

---

## Requirements & Usage

### Requirements

To run the notebooks and replicate the experiments, the following environment is recommended:

- **Python 3.8+**
- Key libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow` (for the neural network exploration)
  - `minisom` (for Self-Organizing Maps visualization)
## Future Work

During this project, the dataset included SMILES strings for each molecule, but their potential was not fully utilized at the time due to limited familiarity with their application. In future iterations:

- **Advanced Feature Extraction with RDKit:**  
  SMILES strings can be leveraged using RDKit to extract richer molecular descriptors such as molecular fingerprints, fragment-based features, and additional topological properties, expanding the dataset’s feature set beyond basic descriptors.

- **Capturing Polarity for Improved Predictions:**  
  While TPSA (Topological Polar Surface Area) was included, it only reflects surface area without fully capturing molecular polarity, which plays a significant role in aqueous solubility. Future work should include extracting direct polarity-related features to better inform the models.

- **Enhanced Model Training:**  
  By enriching the feature space using SMILES-derived descriptors, the dataset can better represent critical chemical properties influencing solubility, potentially improving classification accuracy and model generalization.

These improvements will align the project with advanced practices in **AI-driven materials and chemical property prediction**, providing deeper insights for applications in early-stage molecule screening.

---
