# 神经网络简介

# 学习目标：

#     使用 TensorFlow DNNRegressor 类定义神经网络 (NN) 及其隐藏层
#     训练神经网络学习数据集中的非线性规律，并实现比线性回归模型更好的效果

# 在之前的练习中，我们使用合成特征来帮助模型学习非线性规律。

# 一组重要的非线性关系是纬度和经度的关系，但也可能存在其他非线性关系。

# 现在我们从之前练习中的逻辑回归任务回到标准的（线性）回归任务。也就是说，我们将直接预测 median_house_value。


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
     "longitude"]]
     # "housing_median_age",
     # "total_rooms",
     # "total_bedrooms",
     # "population",
     # "households",
     # "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  # processed_features["rooms_per_person"] = (
  #   california_housing_dataframe["total_rooms"] /
  #   california_housing_dataframe["population"])
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

# 构建神经网络

# 神经网络由 DNNRegressor 类定义。

# 使用 hidden_units 定义神经网络的结构。hidden_units 参数会创建一个整数列表，其中每个整数对应一个隐藏层，表示其中的节点数。
# 以下面的赋值为例：

# hidden_units=[3,10]

# 上述赋值为神经网络指定了两个隐藏层：

#     第一个隐藏层包含 3 个节点。
#     第二个隐藏层包含 10 个节点。

# 如果我们想要添加更多层，可以向该列表添加更多整数。例如，hidden_units=[10,20,30,40] 会创建 4 个分别包含 10、20、30 和 40 个单元的隐藏层。

# 默认情况下，所有隐藏层都会使用 ReLu 激活函数，且是全连接层。


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


def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
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
  # my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  # 还是一样，在这里使用神经网络,层数和节点数直接作为参数传递到DNNRegressor类：
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
  
  # Create input functions
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
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
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

  print ("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print ("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor, training_rmse, validation_rmse

# _ = train_nn_regression_model(
#   my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
#   steps=5000,
#   batch_size=70,
#   hidden_units=[10, 10],
#   training_examples=training_examples,
#   training_targets=training_targets,
#   validation_examples=validation_examples,
#   validation_targets=validation_targets)


# 线性缩放

# 将输入标准化以使其位于 (-1, 1) 范围内可能是一种良好的标准做法。
# 这样一来，SGD 在一个维度中采用很大步长（或者在另一维度中采用很小步长）时不会受阻。
# 数值优化的爱好者可能会注意到，这种做法与使用预调节器 (Preconditioner) 的想法是有联系的。

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)


# 任务 1：使用线性缩放将特征标准化

# 将输入标准化到 (-1, 1) 这一范围内。

# 花费 5 分钟左右的时间来训练和评估新标准化的数据。您能达到什么程度的效果？

# 一般来说，当输入特征大致位于相同范围时，神经网络的训练效果最好。

# 对您的标准化数据进行健全性检查。（如果您忘了将某个特征标准化，会发生什么情况？）

# def normalize_linear_scale(examples_dataframe):     #传入参数examples_dataframe是后面调用函数的时候决定的，在这里是preprocess_features(california_housing_dataframe)
#   """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
#   #
#   # Your code here: normalize the inputs.
#   #
#   preprocess_features=pd.DataFrame()
#   preprocess_features["latitude"]=linear_scale(examples_dataframe["latitude"])
#   preprocess_features["longitude"]=linear_scale(examples_dataframe["longitude"])
#   preprocess_features["housing_median_age"]=linear_scale(examples_dataframe["housing_median_age"])
#   preprocess_features["total_rooms"]=linear_scale(examples_dataframe["total_rooms"])
#   preprocess_features["population"]=linear_scale(examples_dataframe["population"])
#   preprocess_features["households"]=linear_scale(examples_dataframe["households"])
#   preprocess_features["median_income"]=linear_scale(examples_dataframe["median_income"])
#   preprocess_features["rooms_per_person"]=linear_scale(examples_dataframe["rooms_per_person"])
#   return preprocess_features
#   # return linear_scale(examples_dataframe)     #自己尝试写了这一句，但结果并不对，需要一个个特征进行linear_scale
#   # pass

# normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
# # print(normalized_dataframe)
# normalized_training_examples = normalized_dataframe.head(12000)
# normalized_validation_examples = normalized_dataframe.tail(5000)

# _ = train_nn_regression_model(
#     # my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),     #把梯度下降法放到这里来了
#     # my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0003),
#     # my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.003),     
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

# 实验结果：

# 参数：
# learning_rate=0.0007
# steps=5000,
# batch_size=70,
# hidden_units=[10, 10],
# 误差：
# Final RMSE (on training data):   87.91
# Final RMSE (on validation data): 89.56

# 参数：
# learning_rate=0.0003
# steps=5000,
# batch_size=70,
# hidden_units=[10, 10],
# 误差：
# Final RMSE (on training data):   118.92
# Final RMSE (on validation data): 119.90

# 参数：
# learning_rate=0.001
# steps=5000,
# batch_size=70,
# hidden_units=[10, 10],
# 误差：
# Final RMSE (on training data):   77.53
# Final RMSE (on validation data): 75.10

# 参数：
# learning_rate=0.003
# steps=5000,
# batch_size=70,
# hidden_units=[10, 10],
# 误差：
# Final RMSE (on training data):   68.92
# Final RMSE (on validation data): 70.72

# 结论：似乎学习速率升高能够减小损失。

# 参考答案：
# _ = train_nn_regression_model(
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
#     steps=2000,
#     batch_size=50,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

# 结果：
# Final RMSE (on training data):   71.65
# Final RMSE (on validation data): 73.03



# 任务 2：尝试其他优化器

# 使用 AdaGrad 和 Adam 优化器并对比其效果。

# AdaGrad 优化器是一种备选方案。AdaGrad 的核心是灵活地修改模型中每个系数的学习率，从而单调降低有效的学习率。
# 该优化器对于凸优化问题非常有效，但不一定适合非凸优化问题的神经网络训练。
# 您可以通过指定 AdagradOptimizer（而不是 GradientDescentOptimizer）来使用 AdaGrad。
# 请注意，对于 AdaGrad，您可能需要使用较大的学习率。

# 对于非凸优化问题，Adam 有时比 AdaGrad 更有效。
# 要使用 Adam，请调用 tf.train.AdamOptimizer 方法。
# 此方法将几个可选超参数作为参数，但我们的解决方案仅指定其中一个 (learning_rate)。
# 在应用设置中，您应该谨慎指定和调整可选超参数。



# AdaGrad:
# _ = train_nn_regression_model(
# 这里的 adagrad_training_losses和 adagrad_validation_losses就是最后训练集和验证集的RMSE，不定义也没有关系
# _ ,adagrad_training_losses, adagrad_validation_losses= train_nn_regression_model(
#     my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)
# print(adagrad_training_losses)
# print(adagrad_validation_losses)

# 结果：
# Final RMSE (on training data):   106.99
# Final RMSE (on validation data): 107.60

# Adam：
# _ ,adam_training_losses, adam_validation_losses= train_nn_regression_model(
#     my_optimizer=tf.train.AdamOptimizer(learning_rate=0.003),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

# 结果：
# Final RMSE (on training data):   67.55
# Final RMSE (on validation data): 67.38


# 任务 3：尝试其他标准化方法

# 尝试对各种特征使用其他标准化方法，以进一步提高性能。

# 如果仔细查看转换后数据的汇总统计信息，您可能会注意到，对某些特征进行线性缩放会使其聚集到接近 -1 的位置。

# 例如，很多特征的中位数约为 -0.8，而不是 0.0。

_ = training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=2)
 

# 通过选择其他方式来转换这些特征，我们可能会获得更好的效果。

# 例如，对数缩放可能对某些特征有帮助。或者，截取极端值可能会使剩余部分的信息更加丰富。
def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))


# 上述部分包含一些额外的标准化函数。请尝试其中的某些函数，或添加您自己的函数。

# 请注意，如果您将目标标准化，则需要将网络的预测结果非标准化，以便比较损失函数的值。

# 这题是将不同的标准化方法组合起来，构建一个新的 preprocess_features
# def normalize(examples_dataframe):
#   """Returns a version of the input `DataFrame` that has all its features normalized."""
#   #
#   # YOUR CODE HERE: Normalize the inputs.
#   #
#   preprocess_features=pd.DataFrame()
#   preprocess_features["latitude"]=linear_scale(examples_dataframe["latitude"])
#   preprocess_features["longitude"]=linear_scale(examples_dataframe["longitude"])
#   preprocess_features["housing_median_age"]=linear_scale(examples_dataframe["housing_median_age"])

#   # 本身数值较大的用了对数缩放
#   preprocess_features["total_rooms"]=log_normalize(examples_dataframe["total_rooms"])
#   preprocess_features["population"]=log_normalize(examples_dataframe["population"])
#   preprocess_features["households"]=log_normalize(examples_dataframe["households"])

#   # 有较大离群值的，用clip处理
#   preprocess_features["median_income"]=clip(examples_dataframe["median_income"],1,10)
#   preprocess_features["rooms_per_person"]=clip(examples_dataframe["rooms_per_person"],1,30)
#   return preprocess_features
#   # pass

# normalized_dataframe = normalize(preprocess_features(california_housing_dataframe))
# normalized_training_examples = normalized_dataframe.head(12000)
# normalized_validation_examples = normalized_dataframe.tail(5000)

# _ = train_nn_regression_model(
#     # my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.003),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

# 实验结果：
# learning_rate=0.0007：
# Final RMSE (on training data):   78.62
# Final RMSE (on validation data): 78.84

# learning_rate=0.003：
# Final RMSE (on training data):   69.58
# Final RMSE (on validation data): 69.11


# 可选挑战：仅使用纬度和经度特征

# 训练仅使用纬度和经度作为特征的神经网络模型。

# 房地产商喜欢说，地段是房价的唯一重要特征。 我们来看看能否通过训练仅使用纬度和经度作为特征的模型来证实这一点。

# 只有我们的神经网络模型可以从纬度和经度中学会复杂的非线性规律，才能达到我们想要的效果。

# 注意：我们可能需要一个网络结构，其层数比我们之前在练习中使用的要多。


# def normalize(examples_dataframe):
#   """Returns a version of the input `DataFrame` that has all its features normalized."""
#   #
#   # YOUR CODE HERE: Normalize the inputs.
#   #
#   preprocess_features=pd.DataFrame()
#   preprocess_features["latitude"]=linear_scale(examples_dataframe["latitude"])
#   preprocess_features["longitude"]=linear_scale(examples_dataframe["longitude"])
#   return preprocess_features
#   # pass

# normalized_dataframe = normalize(preprocess_features(california_housing_dataframe))
# normalized_training_examples = normalized_dataframe.head(12000)
# normalized_validation_examples = normalized_dataframe.tail(5000)

# _ = train_nn_regression_model(
#     # my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.003),
#     steps=5000,
#     batch_size=70,
#     # hidden_units=[10, 10],
#     hidden_units=[10, 10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

# 实验结果：
# hidden_units=[10, 10]：
# Final RMSE (on training data):   98.94
# Final RMSE (on validation data): 98.10

# hidden_units=[10, 10, 10]:
# Final RMSE (on training data):   97.08
# Final RMSE (on validation data): 100.70

# 参考答案：
def location_location_location(examples_dataframe):
  """Returns a version of the input `DataFrame` that keeps only the latitude and longitude."""
  processed_features = pd.DataFrame()
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  return processed_features

lll_dataframe = location_location_location(preprocess_features(california_housing_dataframe))
lll_training_examples = lll_dataframe.head(12000)
lll_validation_examples = lll_dataframe.tail(5000)

_ = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
    steps=500,
    batch_size=50,
    hidden_units=[10, 10, 5, 5, 5],
    training_examples=lll_training_examples,
    training_targets=training_targets,
    validation_examples=lll_validation_examples,
    validation_targets=validation_targets)

# 实验结果：
# Final RMSE (on training data):   100.31
# Final RMSE (on validation data): 100.23