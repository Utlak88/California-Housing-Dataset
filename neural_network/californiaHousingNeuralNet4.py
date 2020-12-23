# Neural network based on the Google Machine Learning Crash Course

################################################################################
# Importing modules
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

print("Imported modules.")
################################################################################


################################################################################
# Defining functions for model as well as data visualizations
def plot_the_loss_curve(epochs, mse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.show()


print("Defined function to generate curve of loss vs epoch.")


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


def create_model(my_learning_rate, my_feature_layer):
    """Create and compile a simple linear regression model."""

    model = tf.keras.models.Sequential()
    model.add(my_feature_layer)
    model.add(
        tf.keras.layers.Dense(
            units=20,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.04),
            name="Hidden1",
        )
    )

    model.add(
        tf.keras.layers.Dense(
            units=12,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.04),
            name="Hidden2",
        )
    )

    model.add(tf.keras.layers.Dense(units=1, name="Output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )

    return model


def train_model(model, dataset, epochs, label_name, batch_size=None):
    """Train the model by feeding it data."""

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(
        x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True
    )

    epochs = history.epoch

    # Track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse


print("Defined the create_model and train_model functions.")
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
# shuffle the examples
train_data = train_data.reindex(np.random.permutation(train_data.index))
test_data = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
)

print("Imported data.")
################################################################################


################################################################################
# TRAIN FEATURE ENGINEERING (PART 1)
# Defining variable for use to assign column values to column variables
# data = train_data

# # Initially defining column variables
# (
#     longitude,
#     latitude,
#     housing_median_age,
#     total_rooms,
#     total_bedrooms,
#     population,
#     households,
#     median_income,
#     median_house_value,
# ) = range(0, len(data.columns))

# # Assigning column values to column variables
# dict_for_columns = {}
# for x in range(0, len(data.columns)):
#     dict_for_columns[data.columns[x]] = data[data.columns[x]]

# # Defining column variables for use in data analysis
# globals().update(dict_for_columns)

# # Visualizing data
# # train_data.hist(figsize=[20,13])
# # train_data.boxplot(figsize=[20,13])
# # train_data.drop('median_house_value',axis=1).boxplot(figsize=[20,13])

# # Clipping outliers
# total_rooms[total_rooms > 6000] = 6000
# train_data[train_data.columns[3]] = total_rooms

# total_bedrooms[total_bedrooms > 1300] = 1300
# train_data[train_data.columns[4]] = total_bedrooms

# population[population > 3000] = 3000
# train_data[train_data.columns[5]] = population

# households[households > 1250] = 1250
# train_data[train_data.columns[6]] = households

# median_income[median_income > 8.5] = 8.5
# train_data[train_data.columns[7]] = median_income

# print("Clipped train features.")

# Z-Score Normalizing
columns_for_normalizing = train_data[train_data.columns[0:9]]

normalized_columns = (
    columns_for_normalizing - columns_for_normalizing.mean()
) / columns_for_normalizing.std()

train_data[normalized_columns.columns] = normalized_columns

print("Normalized train features.")

# # Revisualizing data
# # train_data.hist(figsize=[20,13])
# # train_data.drop('median_house_value',axis=1).boxplot(figsize=[20,13])

# # Adding new feature calculating the ratio of total bedrooms to total rooms
# train_data["rooms_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]

# print("Added new train data feature calculating the ratio of total bedrooms to total rooms.")
################################################################################


################################################################################
# TEST FEATURE ENGINEERING (PART 1)
# Defining variable for use to assign column values to column variables
# data = test_data

# # Initially defining column variables
# (
#     longitude,
#     latitude,
#     housing_median_age,
#     total_rooms,
#     total_bedrooms,
#     population,
#     households,
#     median_income,
#     median_house_value,
# ) = range(0, len(data.columns))

# # Assigning column values to column variables
# dict_for_columns = {}
# for x in range(0, len(data.columns)):
#     dict_for_columns[data.columns[x]] = data[data.columns[x]]

# # Defining column variables for use in data analysis
# globals().update(dict_for_columns)

# # Visualizing data
# # test_data.hist(figsize=[20,13])
# # test_data.boxplot(figsize=[20,13])
# # test_data.drop('median_house_value',axis=1).boxplot(figsize=[20,13])

# # Clipping outliers
# total_rooms[total_rooms > 6000] = 6000
# test_data[test_data.columns[3]] = total_rooms

# total_bedrooms[total_bedrooms > 1300] = 1300
# test_data[test_data.columns[4]] = total_bedrooms

# population[population > 3000] = 3000
# test_data[test_data.columns[5]] = population

# households[households > 1250] = 1250
# test_data[test_data.columns[6]] = households

# median_income[median_income > 8.5] = 8.5
# test_data[test_data.columns[7]] = median_income

# print("Clipped test features.")

# Z-Score Normalizing
columns_for_normalizing = test_data[test_data.columns[0:9]]

normalized_columns = (
    columns_for_normalizing - columns_for_normalizing.mean()
) / columns_for_normalizing.std()

test_data[normalized_columns.columns] = normalized_columns

print("Normalized test features.")

# # Revisualizing data
# # test_data.hist(figsize=[20,13])
# # test_data.drop('median_house_value',axis=1).boxplot(figsize=[20,13])

# # Adding new feature calculating the ratio of total bedrooms to total rooms
# test_data["rooms_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]

# print("Added new test data feature calculating the ratio of total bedrooms to total rooms.")
################################################################################


################################################################################
# FEATURE ENGINEERING (PART 2)
# Create an empty list that will eventually hold all created feature columns.
feature_columns = []

# Establishing resolution by Zs
resolution_in_Zs = 0.3  # 3/10 of a standard deviation.

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(
    np.arange(
        int(min(train_data["latitude"])),
        int(max(train_data["latitude"])),
        resolution_in_Zs,
    )
)
latitude = tf.feature_column.bucketized_column(
    latitude_as_a_numeric_column, latitude_boundaries
)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(
    np.arange(
        int(min(train_data["longitude"])),
        int(max(train_data["longitude"])),
        resolution_in_Zs,
    )
)
longitude = tf.feature_column.bucketized_column(
    longitude_as_a_numeric_column, longitude_boundaries
)

# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column(
    [latitude, longitude], hash_bucket_size=100
)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

# Convert the list of feature columns into a layer that will later be fed into the model.
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
################################################################################


################################################################################
# TRAINING AND EVALUATING MODEL
# The following variables are the hyperparameters.
learning_rate = 0.005
epochs = 200
batch_size = 1000

label_name = "median_house_value"

# Establish model topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set.
epochs, mse = train_model(my_model, train_data, epochs, label_name, batch_size)
plot_the_loss_curve(epochs, mse)

test_features = {name: np.array(value) for name, value in test_data.items()}
test_label = np.array(test_features.pop(label_name))  # isolate the label
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
################################################################################


################################################################################
# Predicting data using trained model
predicted_values = np.squeeze(my_model.predict(test_features))

print("Predicted data using model.")
################################################################################


################################################################################
# Plotting comparison of predicted to test data in form of R-squared plot
print("Generating R-squared plot to evaluate quality of model prediction of test data.")
plot_r_squared_comparison(
    test_label,
    predicted_values,
    "California Median House Value Prediction Quality\nZ-Score Normalized, Longtitude and Latitude Binned and Crossed",
)
################################################################################
