# California-Housing-Dataset
Machine learning models were developed to predict the median house value feature of a California housing dataset. Three algorithms were used: linear regression, XGBoost, and a TensorFlow/Keras neural network.

Feature engineering was conducted prior to model training, which included Z-score normalization, log scaling, addition of a feature (ratio of total bedrooms to total rooms), feature clipping, and feature binning as well as crossing of longitude and latitude.

For all three algorithms, R-squared graphs were plotted with differing combinations of engineered features to observe effectiveness. These plots were then graphically compiled on an algorithmic basis for ease of comparison.

A legend for each algorithm is provided below linking numerical values to sets of engineered features. The machine learning scripts provided in this repository coincide with the legends, e.g., '1: No Feature Engineering' for XGBoost coincides with 'californiaHousingXGBoost1.py' in the XGBoost folder.

R-squared plots are located in the respective 'plots' folders within each parent algorithm folder. Provided within these are individual .svg graphs for all instances of the feature engineered sets listed in the below legend as well as compilations of the graphs in .xlsx, .docx, .pdf, and .png file formats.

**Linear regression legend:**

- 1:  No Feature Engineering
- 2:  Z-Score Normalized
- 3:  Longitude and Latitude Binned
- 4:  Log Scaled
- 5:  Z-Score Normalized, Feature Added
- 6:  Z-Score Normalized, Features Clipped
- 7:  Z-Score Normalized, Feature Added, Longitude and Latitude Binned
- 8:  Z-Score Normalized, Longitude and Latitude Binned
- 9:  Z-Score Normalized, Features Clipped, Longitude and Latitude Binned
- 10: Z-Score Normalized, Feature Added, Features Clipped, Longitude and Latitude Binned
- 11: Log Scaled, Features Clipped
- 12: Log Scaled, Longitude and Latitude Binned
- 13: Log Scaled, Features Clipped, Longitude and Latitude Binned
- 14: Log Scaled, Feature Added, Features Clipped, Longitude and Latitude Binned


**XGBoost legend:**
- 1:  No Feature Engineering
- 2:  Z-Score Normalized
- 3:  Longitude and Latitude Binned
- 4:  Z-Score Normalized, Log Scaled
- 5:  Z-Score Normalized, Feature Added
- 6:  Z-Score Normalized, Features Clipped
- 7:  Z-Score Normalized, Feature Added, Longitude and Latitude Binned
- 8:  Z-Score Normalized, Feature Added, Features Clipped, Longitude and Latitude Binned
- 9:  Z-Score Normalized, Features Clipped, Longitude and Latitude Binned
- 10: Z-Score Normalized, Longitude and Latitude Binned


**TensorFlow/Keras neural network legend:**
- 1: No Feature Engineering
- 2: Longitude and Latitude Binned and Crossed
- 3: Z-Score Normalized
- 4: Z-Score Normalized, Longitude and Latitude Binned and Crossed
- 5: Z-Score Normalized, Longitude and Latitude Binned and Crossed, Features Clipped
- 6: Z-Score Normalized, Longitude and Latitude Binned and Crossed, Feature Added
- 7: Z-Score Normalized, Longitude and Latitude Binned and Crossed, Feature Added, Features Clipped