################################################################################
# Importing modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print("Imported modules.")
################################################################################


################################################################################
# Defining functions to visualize data comparisons
def plot_scatter_comparison(y_predict, y_test, y_label):
    """Compare predicted to test data in scatter plot form."""
    x_axis_values = np.arange(0, len(y_test))
    plt.figure()
    plt.xlabel("Data Points")
    plt.ylabel(y_label)
    plt.scatter(x_axis_values, y_predict, label="Predicted Data")
    plt.scatter(x_axis_values, y_test, label="Test Data")
    plt.legend()
    plt.show()


print("Defined function to generate scatterplot comparing predicted and test data.")


def plot_r_squared_comparison(y_test, y_predict, title):
    """Produce R-squared plot to evaluate quality of model prediction of test data."""
    r_squared = metrics.r2_score(y_predict, y_test)
    plt.scatter(y_test, y_predict)
    plt.xlabel("Normalized Actual Values")
    plt.ylabel("Normalized Predicted Values")
    plt.title(title)
    plt.plot(
        np.unique(y_test),
        np.poly1d(np.polyfit(y_test, y_predict, 1))(np.unique(y_test)),
    )
    x_r2_label_placement = pd.Series(y_test).median() - 1.2 * pd.Series(y_test).std()
    y_r2_label_placement = (
        pd.Series(y_predict).median() + 3 * pd.Series(y_predict).std()
    )
    plt.text(
        x_r2_label_placement,
        y_r2_label_placement,
        "R-squared = {0:.2f}".format(r_squared),
    )
    plt.show()


print("Defined function to generate R-squared plot.")


def plot_xgboost_feature_importance():
    """Produce feature importance plot using XGBoost."""
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Feature Importance")
    plt.show()


print("Defined plot to visualize feature importance.")
################################################################################


################################################################################
# Adjusting the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{0:1.3f}".format
################################################################################


################################################################################
# Importing data
train_data = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
)

# Shuffle train data
train_data = train_data.reindex(np.random.permutation(train_data.index))

test_data = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
)

print("Imported data.")
################################################################################


################################################################################
# TRAIN FEATURE ENGINEERING
# Initially analyzing data
# train_data.head()
# train_data.describe()
# train_data.info()

# Determining if datasets have missing values
# train_data.isnull().values.any()

# Printing column label indexes
# for (i, item) in enumerate(train_data):
#     print(i, item)

# Defining variable for use to assign column values to column variables
data = train_data

# Initially defining column variables
(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    median_house_value,
) = range(0, len(data.columns))

# Assigning column values to column variables
dict_for_columns = {}
for x in range(0, len(data.columns)):
    dict_for_columns[data.columns[x]] = data[data.columns[x]]

# Defining column variables for use in data analysis
globals().update(dict_for_columns)

# Visualizing data
# train_data.hist(figsize=[20, 13])
# train_data.boxplot(figsize=[20, 13])
# train_data.drop("median_house_value", axis=1).boxplot(figsize=[20, 13])

# Clipping outliers
total_rooms[total_rooms > 6000] = 6000
train_data[train_data.columns[3]] = total_rooms

total_bedrooms[total_bedrooms > 1300] = 1300
train_data[train_data.columns[4]] = total_bedrooms

population[population > 3000] = 3000
train_data[train_data.columns[5]] = population

households[households > 1250] = 1250
train_data[train_data.columns[6]] = households

median_income[median_income > 8.5] = 8.5
train_data[train_data.columns[7]] = median_income

print("Clipped train features.")

# Z-Score Normalizing
# columns_for_normalizing = train_data[train_data.columns[0:9]]

# normalized_columns = (
#     columns_for_normalizing - columns_for_normalizing.mean()
# ) / columns_for_normalizing.std()

# train_data[normalized_columns.columns] = normalized_columns

# print("Normalized train features.")

# Log Scaling
train_data[train_data.columns[3:7]] = np.log(train_data[train_data.columns[3:7]])

print("Log scaled train features.")

# Revisualizing data
# train_data.hist(figsize=[20,13])
# train_data.drop('median_house_value',axis=1).boxplot(figsize=[20,13])

# Adding new feature calculating the ratio of total bedrooms to total rooms
train_data["rooms_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]

print(
    "Added new train data feature calculating the ratio of total bedrooms to total rooms."
)

# BINNING LONGITUDE AND LATITUDE
# Longitude binning
res_long, bins_long = pd.qcut(train_data["longitude"], q=10, retbins=True)

which_longitude_bin = np.digitize(
    np.reshape(train_data["longitude"].to_numpy(), (len(train_data["longitude"]), 1)),
    bins=bins_long,
)

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_longitude_bin)

longitude_binned = encoder.transform(which_longitude_bin)

# Latitude binning
res_lat, bins_lat = pd.qcut(train_data["latitude"], q=10, retbins=True)

which_latitude_bin = np.digitize(
    np.reshape(train_data["latitude"].to_numpy(), (len(train_data["latitude"]), 1)),
    bins=bins_lat,
)

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_latitude_bin)

latitude_binned = encoder.transform(which_latitude_bin)

# Appending longitude one hot encoded columns
bins_long_index_list = []

for i in range(bins_long.size):
    bins_long_index_list.append(str(i) + "_long")
    i += 1

train_data[bins_long_index_list] = longitude_binned

# Appending latitude one hot encoded columns
bins_lat_index_list = []

for i in range(bins_lat.size):
    bins_lat_index_list.append(str(i) + "_lat")
    i += 1

train_data[bins_lat_index_list] = latitude_binned

print("Binned train longitude and latitude.")
################################################################################


################################################################################
# TEST FEATURE ENGINEERING
# Initially analyzing data
# test_data.head()
# test_data.describe()
# test_data.info()

# Determining if datasets have missing values
# test_data.isnull().values.any()

# Printing column label indexes
# for (i, item) in enumerate(test_data):
#     print(i, item)

# Defining variable for use to assign column values to column variables
data = test_data

# Initially defining column variables
(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    median_house_value,
) = range(0, len(data.columns))

# Assigning column values to column variables
dict_for_columns = {}
for x in range(0, len(data.columns)):
    dict_for_columns[data.columns[x]] = data[data.columns[x]]

# Defining column variables for use in data analysis
globals().update(dict_for_columns)

# Visualizing data
# test_data.hist(figsize=[20, 13])
# test_data.boxplot(figsize=[20, 13])
# test_data.drop("median_house_value", axis=1).boxplot(figsize=[20, 13])

# Clipping outliers
total_rooms[total_rooms > 6000] = 6000
test_data[test_data.columns[3]] = total_rooms

total_bedrooms[total_bedrooms > 1300] = 1300
test_data[test_data.columns[4]] = total_bedrooms

population[population > 3000] = 3000
test_data[test_data.columns[5]] = population

households[households > 1250] = 1250
test_data[test_data.columns[6]] = households

median_income[median_income > 8.5] = 8.5
test_data[test_data.columns[7]] = median_income

print("Clipped test features.")

# Z-Score Normalizing
# columns_for_normalizing = test_data[test_data.columns[0:9]]

# normalized_columns = (
#     columns_for_normalizing - columns_for_normalizing.mean()
# ) / columns_for_normalizing.std()

# test_data[normalized_columns.columns] = normalized_columns

# print("Normalized test features.")

# Log Scaling
test_data[test_data.columns[3:7]] = np.log(test_data[test_data.columns[3:7]])

print("Log scaled test features.")

# Revisualizing data
# test_data.hist(figsize=[20,13])
# test_data.drop('median_house_value',axis=1).boxplot(figsize=[20,13])

# Adding new feature calculating the ratio of total bedrooms to total rooms
test_data["rooms_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]

print(
    "Added new test data feature calculating the ratio of total bedrooms to total rooms."
)

# BINNING LONGITUDE AND LATITUDE
# Longitude binning
res_long, bins_long = pd.qcut(test_data["longitude"], q=10, retbins=True)

which_longitude_bin = np.digitize(
    np.reshape(test_data["longitude"].to_numpy(), (len(test_data["longitude"]), 1)),
    bins=bins_long,
)

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_longitude_bin)

longitude_binned = encoder.transform(which_longitude_bin)

# Latitude binning
res_lat, bins_lat = pd.qcut(test_data["latitude"], q=10, retbins=True)

which_latitude_bin = np.digitize(
    np.reshape(test_data["latitude"].to_numpy(), (len(test_data["latitude"]), 1)),
    bins=bins_lat,
)

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_latitude_bin)

latitude_binned = encoder.transform(which_latitude_bin)

# Appending longitude one hot encoded columns
bins_long_index_list = []

for i in range(bins_long.size):
    bins_long_index_list.append(str(i) + "_long")
    i += 1

test_data[bins_long_index_list] = longitude_binned

# Appending latitude one hot encoded columns
bins_lat_index_list = []

for i in range(bins_lat.size):
    bins_lat_index_list.append(str(i) + "_lat")
    i += 1

test_data[bins_lat_index_list] = latitude_binned

print("Binned test longitude and latitude.")
################################################################################


################################################################################
# Defining model features and target
train_data_features = train_data.drop(["median_house_value"], axis=1)
test_data_features = test_data.drop(["median_house_value"], axis=1)

train_data_target = train_data["median_house_value"]
test_data_target = test_data["median_house_value"]

print("Defined model features and target.")
################################################################################


################################################################################
# Establishing model topography.
model = LinearRegression()

print("Established model topology.")
################################################################################


################################################################################
# Training model
# model().fit(data_features_train, pd.DataFrame(data_target_train))
model.fit(train_data_features, train_data_target)

print("Trained model.")
################################################################################


################################################################################
# Predicting data using trained model
predicted_target_feature = model.predict(test_data_features)

print("Predicted data using model.\n")
################################################################################


################################################################################
# Plotting comparison of predicted to test data in form of R-squared plot
print("Generating R-squared plot to evaluate quality of model prediction of test data.")

# plot_r_squared_comparison(data_target_test, predicted_target_feature, 'California Median House Value Prediction Quality\nLog Scaled, Feature Added, Features Clipped, Longitude and Latitude Binned')


plot_r_squared_comparison(
    test_data_target,
    predicted_target_feature,
    "California Median House Value Prediction Quality\nLog Scaled, Feature Added, Features Clipped,\nLongitude and Latitude Binned",
)
################################################################################
