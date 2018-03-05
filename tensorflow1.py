# 例1.TensorFlow基础:

# 1.加载必要的第三方库：
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

# 2.加载数据集：
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")

# 3.我们将对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。
# 此外，我们会将 median_house_value 调整为以千为单位，这样，模型就能够以常用范围内的学习速率较为轻松地学习这些数据。
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))		#用来随机化处理数据的代码。几乎任何数据集都要经过随机化预处理，来保证有效性，这句代码非常重要！！				
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe

# 4.检查数据：
# print(california_housing_dataframe.describe())		#输出关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数。

# 接下来开始构建训练模型：


# 1.构建特征列：
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]
# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]			#使用 numeric_column 定义特征列，这样会将其数据指定为数值：

# 2.定义标签(希望被预测的目标)，在本例中选择了median_house_value：
# Define the label.
targets = california_housing_dataframe["median_house_value"]

# 3.配置 LinearRegressor：
# 线性回归的算法本身LinearRegressor是TensorFlow自带的，我们要做的，是给定线性回归的特征列和优化算法（梯度下降）
# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)		#使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）训练该模型。learning_rate 参数可控制梯度步长的大小
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# 为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

# 4.定义输入函数：
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a linear regression model of one feature.
	熟悉以下注释内容，了解输入函数各参数的含义。  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      
      shuffle: True or False. Whether to shuffle the data.
      如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。
      buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。
      
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}		#类似两个库之间数据类型的转换：从panda把数据集转到numpy数组                                 
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)								#这里默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
    # 使用 TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复。

    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
    # 输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。

# 5.训练模型：
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)
# # 现在，我们可以在 linear_regressor 上调用 train() 来训练模型。
# # 我们会将 my_input_fn 封装在匿名函数lambda 中，以便可以将 my_feature 和 target 作为参数传入。
# # 首先，我们会训练 100 步。

# # 6.评估模型效果：
# # Create an input function for predictions.
# # Note: Since we're making just one prediction for each example, we don't 
# # need to repeat or shuffle the data here.
# prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# # Call predict() on the linear_regressor to make predictions.
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# # 预测结果调用 linear_regressor的 predict()

# # Format predictions as a NumPy array, so we can calculate error metrics.
# predictions = np.array([item['predictions'][0] for item in predictions])

# # Print Mean Squared Error and Root Mean Squared Error.
# mean_squared_error = metrics.mean_squared_error(predictions, targets)
# root_mean_squared_error = math.sqrt(mean_squared_error)
# # MSE和RMSE都是自带的
# # print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
# # print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

# # 7.如何缩小误差？
# # 首先，我们可以了解一下根据总体摘要统计信息，预测和目标的符合情况:
# calibration_data = pd.DataFrame()
# calibration_data["predictions"] = pd.Series(predictions)
# calibration_data["targets"] = pd.Series(targets)
# print(calibration_data.describe())

# # 我们还可以将数据和学到的线可视化。我们已经知道，单个特征的线性回归可绘制成一条将输入 x 映射到输出 y 的线。

# # 首先，我们将获得均匀分布的随机数据样本，以便绘制可辨的散点图。
# sample = california_housing_dataframe.sample(n=300)

# # 然后，我们根据模型的偏差项和特征权重绘制学到的线，并绘制散点图。该线会以红色显示:
# # Get the min and max total_rooms values.
# x_0 = sample["total_rooms"].min()
# x_1 = sample["total_rooms"].max()

# # Retrieve the final weight and bias generated during training.
# weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
# bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# # Get the predicted median_house_values for the min and max total_rooms values.
# y_0 = weight * x_0 + bias 
# y_1 = weight * x_1 + bias

# # Plot our regression line from (x_0, y_0) to (x_1, y_1).
# plt.plot([x_0, x_1], [y_0, y_1], c='r')

# # Label the graph axes.
# plt.ylabel("median_house_value")
# plt.xlabel("total_rooms")

# # Plot a scatter plot from our data sample.
# plt.scatter(sample["total_rooms"], sample["median_house_value"])

# # Display graph.
# plt.show()







# 接下来为了方便使用，将以上所有代码重新编入一个函数中：
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # Create feature columns
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
  
  # Create input functions
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print ("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print ("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print( "Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print( "Final RMSE (on training data): %0.2f" % root_mean_squared_error)


# 开始训练模型，设置初始超参数为0.0001，100，1：
# train_model(
#     learning_rate=0.00001,
#     steps=100,
#     batch_size=1
# )

#修正超参数，尝试将RMSE降低到180以内：
# train_model(
#     learning_rate=0.00002,
#     steps=500,
#     batch_size=5
# )

# 有适用于模型调整的标准启发法吗？
# 这是一个常见的问题。简短的答案是，不同超参数的效果取决于数据。因此，不存在必须遵循的规则，您需要对自己的数据进行测试。
# 即便如此，我们仍在下面列出了几条可为您提供指导的经验法则：
#     训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。
#     如果训练尚未收敛，尝试运行更长的时间。
#     如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。
#         但有时如果学习速率过高，训练误差的减小速度反而会变慢。
#     如果训练误差变化很大，尝试降低学习速率。
#         较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。
#     批量大小过小也会导致不稳定情况。不妨先尝试 100 或 1000 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。
# 重申一下，切勿严格遵循这些经验法则，因为效果取决于数据。请始终进行试验和验证。


# 使用 population 特征替换 total_rooms 特征，看看能否取得更好的效果:
train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=1,
    input_feature='population'			#给函数中的input_feature重新赋值
)
#效果好像没有total_rooms 特征好。