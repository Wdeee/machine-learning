import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


def preprocess_features(california_housing_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets



# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double-check that we've done the right thing.
# print ("Training examples summary:")
# display.display(training_examples.describe())
# print ("Validation examples summary:")
# display.display(validation_examples.describe())

# print ("Training targets summary:")
# display.display(training_targets.describe())
# print ("Validation targets summary:")
# display.display(validation_targets.describe())



def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])



def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels



# FTRL 优化算法

# 高维度线性模型可受益于使用一种基于梯度的优化方法，叫做 FTRL。
# 该算法的优势是针对不同系数以不同方式调整学习速率，如果某些特征很少采用非零值，该算法可能比较实用（也非常适合支持 L1 正则化）。
# 我们可以使用 FtrlOptimizer 来应用 FTRL。

# 训练函数代码就是换了算法，其他内容没有差别
def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    feature_columns: A `set` specifying the input feature columns to use.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)				#和之前采用的训练算法不同，不再是梯度下降法了
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print ("Training model...")
  print ("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print ("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print ("Model training finished.")

  
  # Output a graph of loss metrics over periods.
  # plt.ylabel("RMSE")
  # plt.xlabel("Periods")
  # plt.title("Root Mean Squared Error vs. Periods")
  # plt.tight_layout()
  # plt.plot(training_rmse, label="training")
  # plt.plot(validation_rmse, label="validation")
  # plt.legend()
  # plt.show()

  return linear_regressor



# _ = train_model(
#     learning_rate=1.0,
#     steps=500,
#     batch_size=100,
#     feature_columns=construct_feature_columns(training_examples),
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)




# 分桶（分箱）特征

# 分桶也称为分箱。

# 例如，我们可以将 population 分为以下 3 个分桶：

#     bucket_0 (< 5000)：对应于人口分布较少的街区
#     bucket_1 (5000 - 25000)：对应于人口分布适中的街区
#     bucket_2 (> 25000)：对应于人口分布较多的街区

# 根据前面的分桶定义，以下 population 矢量：

# [[10001], [42004], [2500], [18000]]

# 将变成以下经过分桶的特征矢量：

# [[1], [2], [0], [1]]

# 这些特征值现在是分桶索引。请注意，这些索引被视为离散特征。通常情况下，这些特征将被进一步转换为上述独热表示法，但这是以透明方式实现的。

# 要为分桶特征定义特征列，我们可以使用 bucketized_column（而不是使用 numeric_column），该列将数字列作为输入，并使用 boundardies 参数中指定的分桶边界将其转换为分桶特征。
# 以下代码为 households 和 longitude 定义了分桶特征列；get_quantile_based_boundaries 函数会根据分位数计算边界，以便每个分桶包含相同数量的元素。

def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

# Divide households into 7 buckets.
households = tf.feature_column.numeric_column("households")
bucketized_households = tf.feature_column.bucketized_column(
  households, boundaries=get_quantile_based_boundaries(
    california_housing_dataframe["households"], 7))

# Divide longitude into 10 buckets.
longitude = tf.feature_column.numeric_column("longitude")
bucketized_longitude = tf.feature_column.bucketized_column(
  longitude, boundaries=get_quantile_based_boundaries(
    california_housing_dataframe["longitude"], 10))



# 任务 1：使用分桶特征列训练模型

# 将我们示例中的所有实值特征进行分桶，训练模型，然后查看结果是否有所改善。

# 在前面的代码块中，两个实值列（即 households 和 longitude）已被转换为分桶特征列。您的任务是对其余的列进行分桶，然后运行代码来训练模型。
# 您可以采用各种启发法来确定分桶的范围。本练习使用了分位数技巧，通过这种方式选择分桶边界后，每个分桶将包含相同数量的样本。

# def construct_feature_columns():
#   """Construct the TensorFlow Feature Columns.

#   Returns:
#     A set of feature columns
#   """ 
#   households = tf.feature_column.numeric_column("households")
#   longitude = tf.feature_column.numeric_column("longitude")
#   latitude = tf.feature_column.numeric_column("latitude")
#   housing_median_age = tf.feature_column.numeric_column("housing_median_age")
#   median_income = tf.feature_column.numeric_column("median_income")
#   rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
#   # Divide households into 7 buckets.
#   bucketized_households = tf.feature_column.bucketized_column(
#     households, boundaries=get_quantile_based_boundaries(
#       training_examples["households"], 7))

#   # Divide longitude into 10 buckets.
#   bucketized_longitude = tf.feature_column.bucketized_column(
#     longitude, boundaries=get_quantile_based_boundaries(
#       training_examples["longitude"], 10))


# #### my codes here	####
#   bucketized_latitude = tf.feature_column.bucketized_column(
#     latitude, boundaries=get_quantile_based_boundaries(
#       training_examples["latitude"], 10))

#   bucketized_housing_median_age = tf.feature_column.bucketized_column(
#     housing_median_age, boundaries=get_quantile_based_boundaries(
#       training_examples["housing_median_age"], 17))

#   bucketized_median_income =tf.feature_column.bucketized_column(
#     median_income, boundaries=get_quantile_based_boundaries(
#       training_examples["median_income"], 15))

# # rooms_per_person这个参数有一个很大的离群值，不知道怎么分箱比较合适
#   bucketized_rooms_per_person =tf.feature_column.bucketized_column(
#     rooms_per_person, boundaries=get_quantile_based_boundaries(
#       training_examples["rooms_per_person"], 5))
  
#   feature_columns = set([
#     bucketized_longitude,
#     bucketized_latitude,
#     bucketized_housing_median_age,
#     bucketized_households,
#     bucketized_median_income,
#     bucketized_rooms_per_person])
  
#   return feature_columns



# _ = train_model(
#     learning_rate=1.0,
#     steps=500,
#     batch_size=100,
#     feature_columns=construct_feature_columns(),
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

# 我的结论：
# 最终训练误差为：period 09 : 89.62

# 参考答案：

# def construct_feature_columns():
#   """Construct the TensorFlow Feature Columns.

#   Returns:
#     A set of feature columns
#   """ 
#   households = tf.feature_column.numeric_column("households")
#   longitude = tf.feature_column.numeric_column("longitude")
#   latitude = tf.feature_column.numeric_column("latitude")
#   housing_median_age = tf.feature_column.numeric_column("housing_median_age")
#   median_income = tf.feature_column.numeric_column("median_income")
#   rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
#   # Divide households into 7 buckets.
#   bucketized_households = tf.feature_column.bucketized_column(
#     households, boundaries=get_quantile_based_boundaries(
#       training_examples["households"], 7))

#   # Divide longitude into 10 buckets.
#   bucketized_longitude = tf.feature_column.bucketized_column(
#     longitude, boundaries=get_quantile_based_boundaries(
#       training_examples["longitude"], 10))
  
#   # Divide latitude into 10 buckets.
#   bucketized_latitude = tf.feature_column.bucketized_column(
#     latitude, boundaries=get_quantile_based_boundaries(
#       training_examples["latitude"], 10))

#   # Divide housing_median_age into 7 buckets.
#   bucketized_housing_median_age = tf.feature_column.bucketized_column(
#     housing_median_age, boundaries=get_quantile_based_boundaries(
#       training_examples["housing_median_age"], 7))
  
#   # Divide median_income into 7 buckets.
#   bucketized_median_income = tf.feature_column.bucketized_column(
#     median_income, boundaries=get_quantile_based_boundaries(
#       training_examples["median_income"], 7))
  
#   # Divide rooms_per_person into 7 buckets.
#   bucketized_rooms_per_person = tf.feature_column.bucketized_column(
#     rooms_per_person, boundaries=get_quantile_based_boundaries(
#       training_examples["rooms_per_person"], 7))
  
#   feature_columns = set([
#     bucketized_longitude,
#     bucketized_latitude,
#     bucketized_housing_median_age,
#     bucketized_households,
#     bucketized_median_income,
#     bucketized_rooms_per_person])
  
#   return feature_columns


# _ = train_model(
#     learning_rate=1.0,
#     steps=500,
#     batch_size=100,
#     feature_columns=construct_feature_columns(),
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

# 最终误差：period 09 : 88.45，效果更好。

# 特征组合

# 组合两个（或更多个）特征是使用线性模型来学习非线性关系的一种聪明做法。
# 在我们的问题中，如果我们只使用 latitude 特征进行学习，那么该模型可能会发现特定纬度（或特定纬度范围内，因为我们已经将其分桶）的城市街区更可能比其他街区住房成本高昂。
# longitude 特征的情况与此类似。但是，如果我们将 longitude 与 latitude 组合，产生的组合特征则代表一个明确的城市街区。如果模型发现某些城市街区（位于特定纬度和经度范围内）更可能比其他街区住房成本高昂，那么这将是比单独考虑两个特征更强烈的信号。

# 目前，特征列 API 仅支持组合离散特征。要组合两个连续的值（比如 latitude 或 longitude），我们可以对其进行分桶。

# 如果我们组合 latitude 和 longitude 特征（例如，假设 longitude 被分到 2 个分桶中，而 latitude 有 3 个分桶），我们实际上会得到 6 个组合的二元特征。
# 当我们训练模型时，每个特征都会分别获得自己的权重。


# 任务 2：使用特征组合训练模型

# 在模型中添加 longitude 与 latitude 的特征组合，训练模型，然后确定结果是否有所改善。

# 请参阅有关 crossed_column() 的 TensorFlow API 文档，了解如何为您的组合构建特征列。hash_bucket_size 可以设为 1000。

def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 
  households = tf.feature_column.numeric_column("households")
  longitude = tf.feature_column.numeric_column("longitude")
  latitude = tf.feature_column.numeric_column("latitude")
  housing_median_age = tf.feature_column.numeric_column("housing_median_age")
  median_income = tf.feature_column.numeric_column("median_income")
  rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
  # Divide households into 7 buckets.
  bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(
      training_examples["households"], 7))

  # Divide longitude into 10 buckets.
  bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_boundaries(
      training_examples["longitude"], 10))
  
  # Divide latitude into 10 buckets.
  bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_boundaries(
      training_examples["latitude"], 10))

  # Divide housing_median_age into 7 buckets.
  bucketized_housing_median_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=get_quantile_based_boundaries(
      training_examples["housing_median_age"], 7))
  
  # Divide median_income into 7 buckets.
  bucketized_median_income = tf.feature_column.bucketized_column(
    median_income, boundaries=get_quantile_based_boundaries(
      training_examples["median_income"], 7))
  
  # Divide rooms_per_person into 7 buckets.
  bucketized_rooms_per_person = tf.feature_column.bucketized_column(
    rooms_per_person, boundaries=get_quantile_based_boundaries(
      training_examples["rooms_per_person"], 7))
  
  # YOUR CODE HERE: Make a feature column for the long_x_lat feature cross
  # long_x_lat = crossed_column(["bucketized_latitude","bucketized_longitude"],1000)			#自己写的代码，用法错误
  long_x_lat = tf.feature_column.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)
  
  feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person,
    long_x_lat])
  
  return feature_columns



_ = train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# 结论：最终误差：period 09 : 79.03
# 注意crossed_column的调用方法。另外是set()