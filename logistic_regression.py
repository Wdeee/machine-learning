# 逻辑回归

#  学习目标：
#     将（在之前的练习中构建的）房屋价值中位数预测模型重新构建为二元分类模型
#     比较逻辑回归与线性回归解决二元分类问题的有效性

# 与在之前的练习中一样，我们将使用加利福尼亚州住房数据集，
# 但这次我们会预测某个城市街区的住房成本是否高昂，从而将其转换成一个二元分类问题。
# 此外，我们还会暂时恢复使用默认特征。

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


# 注意以下代码与之前练习中的代码之间稍有不同。我们并没有将 median_house_value 用作目标，而是创建了一个新的二元目标 median_house_value_is_high。
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

# target换了一个
def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Create a boolean categorical feature representing whether the
  # medianHouseValue is above a set threshold.
  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)			#通过一个判断语句返回一个布尔值，再用astype命令将布尔值转换为浮点数类型(1.0/0.0)。265000为阈值
  return output_targets


# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double-check that we've done the right thing.
# print( "Training examples summary:")
# display.display(training_examples.describe())
# print ("Validation examples summary:")
# display.display(validation_examples.describe())

# print ("Training targets summary:")
# display.display(training_targets.describe())
# print ("Validation targets summary:")
# display.display(validation_targets.describe())



# 线性回归会有怎样的表现？

# 为了解逻辑回归为什么有效，我们首先训练一个使用线性回归的简单模型。该模型将使用 {0, 1} 中的值为标签，并尝试预测一个尽可能接近 0 或 1 的连续值。
# 此外，我们希望将输出解读为概率，所以最好模型的输出值可以位于 (0, 1) 范围内。然后我们会应用阈值 0.5，以确定标签。

# 运行以下单元格，以使用 LinearRegressor 训练线性回归模型。

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


# def train_linear_regressor_model(
#     learning_rate,
#     steps,
#     batch_size,
#     training_examples,
#     training_targets,
#     validation_examples,
#     validation_targets):
#   """Trains a linear regression model.
  
#   In addition to training, this function also prints training progress information,
#   as well as a plot of the training and validation loss over time.
  
#   Args:
#     learning_rate: A `float`, the learning rate.
#     steps: A non-zero `int`, the total number of training steps. A training step
#       consists of a forward and backward pass using a single batch.
#     batch_size: A non-zero `int`, the batch size.
#     training_examples: A `DataFrame` containing one or more columns from
#       `california_housing_dataframe` to use as input features for training.
#     training_targets: A `DataFrame` containing exactly one column from
#       `california_housing_dataframe` to use as target for training.
#     validation_examples: A `DataFrame` containing one or more columns from
#       `california_housing_dataframe` to use as input features for validation.
#     validation_targets: A `DataFrame` containing exactly one column from
#       `california_housing_dataframe` to use as target for validation.
      
#   Returns:
#     A `LinearRegressor` object trained on the training data.
#   """

#   periods = 10
#   steps_per_period = steps / periods

#   # Create a linear regressor object.
#   my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#   my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
#   linear_regressor = tf.estimator.LinearRegressor(
#       feature_columns=construct_feature_columns(training_examples),
#       optimizer=my_optimizer
#   )
    
#   # Create input functions  
#   training_input_fn = lambda: my_input_fn(training_examples, 
#                                           training_targets["median_house_value_is_high"], 
#                                           batch_size=batch_size)
#   predict_training_input_fn = lambda: my_input_fn(training_examples, 
#                                                   training_targets["median_house_value_is_high"], 
#                                                   num_epochs=1, 
#                                                   shuffle=False)
#   predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
#                                                     validation_targets["median_house_value_is_high"], 
#                                                     num_epochs=1, 
#                                                     shuffle=False)

#   # Train the model, but do so inside a loop so that we can periodically assess
#   # loss metrics.
#   print ("Training model...")
#   print ("RMSE (on training data):")
#   training_rmse = []
#   validation_rmse = []
#   for period in range (0, periods):
#     # Train the model, starting from the prior state.
#     linear_regressor.train(
#         input_fn=training_input_fn,
#         steps=steps_per_period
#     )
    
#     # Take a break and compute predictions.
#     training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
#     training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
#     validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
#     validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
#     # Compute training and validation loss.
#     training_root_mean_squared_error = math.sqrt(
#         metrics.mean_squared_error(training_predictions, training_targets))
#     validation_root_mean_squared_error = math.sqrt(
#         metrics.mean_squared_error(validation_predictions, validation_targets))
#     # Occasionally print the current loss.
#     print ("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
#     # Add the loss metrics from this period to our list.
#     training_rmse.append(training_root_mean_squared_error)
#     validation_rmse.append(validation_root_mean_squared_error)
#   print ("Model training finished.")
  
#   # Output a graph of loss metrics over periods.
#   # plt.ylabel("RMSE")
#   # plt.xlabel("Periods")
#   # plt.title("Root Mean Squared Error vs. Periods")
#   # plt.tight_layout()
#   # plt.plot(training_rmse, label="training")
#   # plt.plot(validation_rmse, label="validation")
#   # plt.legend()
#   # plt.show()
  
#   # _ = plt.hist(validation_predictions)
#   # plt.show()

#   return linear_regressor


# linear_regressor = train_linear_regressor_model(
#     learning_rate=0.000001,
#     steps=200,
#     batch_size=20,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)



# 任务 1：我们可以计算这些预测的对数损失函数吗？

# 检查预测，并确定是否可以使用它们来计算对数损失函数。

# LinearRegressor 使用的是 L2 损失，在将输出解读为概率时，它并不能有效地惩罚误分类。
# 例如，对于概率分别为 0.9 和 0.9999 的负分类样本是否被分类为正分类，二者之间的差异应该很大，但 L2 损失并不会明显区分这些情况。

# 相比之下，LogLoss（对数损失函数）对这些"置信错误"的惩罚力度更大。请注意，LogLoss 的定义如下：

# LogLoss=∑(x,y)∈D−y⋅log(ypred)−(1−y)⋅log(1−ypred)

# 但我们首先需要获得预测值。我们可以使用 LinearRegressor.predict 获得预测值。

# 我们可以使用预测和相应目标计算 LogLoss 吗？

# 参考答案：
# 这道题的答案不是很理解。。。predict_validation_input_fn的定义没有任何改变，只是加了一个画直方图的hist函数
# predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
#                                                   validation_targets["median_house_value_is_high"], 
#                                                   num_epochs=1, 
#                                                   shuffle=False)

# validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
# validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

# _ = plt.hist(validation_predictions)



# 任务 2：训练逻辑回归模型并计算验证集的对数损失函数

# 要使用逻辑回归非常简单，用 LinearClassifier 替代 LinearRegressor 即可。完成以下代码。

# 注意：在 LinearClassifier 模型上运行 train() 和 predict() 时，您可以通过返回的字典（例如 predictions["probabilities"]）中的 "probabilities" 键获取实值预测概率。
# Sklearn 的 log_loss 函数可基于这些概率计算对数损失函数，非常方便

def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of one feature.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  # 参照linear_regressor的代码，把这里换掉即可
  linear_classifier = tf.estimator.LinearClassifier(
  	feature_columns=construct_feature_columns(training_examples),
  	optimizer=my_optimizer)

  # Create input functions
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value_is_high"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value_is_high"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value_is_high"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print ("Training model...")
  print ("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.    
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print ("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print ("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  # plt.ylabel("LogLoss")
  # plt.xlabel("Periods")
  # plt.title("LogLoss vs. Periods")
  # plt.tight_layout()
  # plt.plot(training_log_losses, label="training")
  # plt.plot(validation_log_losses, label="validation")
  # plt.legend()
  # plt.show()

  # 任务 3：计算准确率并为验证集绘制 ROC 曲线

  # 分类时非常有用的一些指标包括：模型准确率、ROC 曲线和 ROC 曲线下面积 (AUC)。我们会检查这些指标。

  # LinearClassifier.evaluate 可计算准确率和 AUC 等实用指标。

  evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

  print ("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
  print ("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
  # 您可以使用类别概率（例如由 LinearClassifier.predict 和 Sklearn 的 roc_curve 计算的概率）来获得绘制 ROC 曲线所需的真正例率和假正例率。
  validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
  # Get just the probabilities for the positive class
  validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

  false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
  plt.plot(false_positive_rate, true_positive_rate, label="our model")
  plt.plot([0, 1], [0, 1], label="random classifier")
  _ = plt.legend(loc=2)
  plt.show()


  return linear_classifier



# 看看您是否可以调整任务 2 中训练的模型的学习设置，以改善 AUC。

# 通常情况下，某些指标在提升的同时会损害其他指标，因此您需要找到可以实现理想折中情况的设置。

# 验证所有指标是否同时有所提升。

# TUNE THE SETTINGS BELOW TO IMPROVE AUC
# linear_classifier = train_linear_classifier_model(
#     learning_rate=0.000005,
#     # learning_rate=0.0005,
#     # learning_rate=0.000001,

#     # steps=500,
#     steps=800,
#     # steps=1000,
    
#     batch_size=20,
#     # batch_size=40,
#     # batch_size=10,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)


# 结论：
# 修改 learning_rate:
# learning_rate=0.000005:
# 	LogLoss:period 09 : 0.53
# 	AUC on the validation set: 0.76
# 	Accuracy on the validation set: 0.77
# learning_rate=0.0005:
# 	 period 09 : 3.25(已经不收敛了，学习速率太高)
# 	 AUC on the validation set: 0.51
# 	 Accuracy on the validation set: 0.76
# learning_rate=0.000001:
# 	LogLoss:period 09 : 0.57
# 	AUC on the validation set: 0.58
# 	Accuracy on the validation set: 0.74

# 修改 steps:
# steps=500:
# 	LogLoss:period 09 : 0.53
# 	AUC on the validation set: 0.76
# 	Accuracy on the validation set: 0.77
# steps=800:
# 	LogLoss:period 09 : 0.52
# 	AUC on the validation set: 0.76
# 	Accuracy on the validation set: 0.77
# steps=1000:
# 	LogLoss:period 09 : 0.52
# 	AUC on the validation set: 0.75
# 	Accuracy on the validation set: 0.77

# 修改 batch_size:
# batch_size=20:
# 	LogLoss:period 09 : 0.52
# 	AUC on the validation set: 0.76
# 	Accuracy on the validation set: 0.77
# batch_size=40:
# 	LogLoss:period 09 : 0.52
# 	AUC on the validation set: 0.75
# 	Accuracy on the validation set: 0.77
# batch_size=10:
# 	LogLoss:period 09 : 0.54
# 	AUC on the validation set: 0.70
# 	Accuracy on the validation set: 0.76

# 综上，选择learning_rate=0.000005，steps=800，batch_size=20

# 参考答案：
#  一个可能有用的解决方案是，只要不过拟合，就训练更长时间。

# 要做到这一点，我们可以增加步数和/或批量大小。

# 所有指标同时提升，这样，我们的损失指标就可以很好地代理 AUC 和准确率了。

# 注意它是如何进行很多很多次迭代，只是为了再尽量增加一点 AUC。
# 这种情况很常见，但通常情况下，即使只有一点小小的收获，投入的成本也是值得的。
linear_classifier = train_linear_classifier_model(
    learning_rate=0.000003,
    steps=20000,
    batch_size=500,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

  # period 09 : 0.47
  # AUC on the validation set: 0.81
  # Accuracy on the validation set: 0.78