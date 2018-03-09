# 例3.特征集

# 学习目标：创建一个包含极少特征但效果与更复杂的特征集一样出色的集合

# 到目前为止，我们已经将所有特征添加到了模型中。具有较少特征的模型会使用较少的资源，并且更易于维护。
# 我们来看看能否构建这样一种模型：包含极少的住房特征，但效果与使用数据集中所有特征的模型一样出色。


# 加载数据：
import math

# import gc

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

# 以下是自己用笨办法写的分箱函数。原理可能是对的,但是循环次数太多，binning_values数据超出了内存上限，所以跑程序时会报MemoryError。
# def binning(latitude):
# 	a=[]
# 	b=[]
# 	for x in latitude:
# 		if x>32.3 and x<=33.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>33.3 and x<=34.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>34.3 and x<=35.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>35.2 and x<=36.3:
# 			a.append(1)
# 		else:
# 			a.append(0)	
# 		if x>36.3 and x<=37.3:
# 			a.append(1)
# 		else:
# 			a.append(0)	
# 		if x>37.3 and x<=38.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>38.3 and x<=39.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>39.3 and x<=40.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>40.3 and x<=41.3:
# 			a.append(1)
# 		else:
# 			a.append(0)
# 		if x>41.3 and x<=42.3:
# 			a.append(1)
# 		else:
# 			a.append(0)	
# 		b.append(a)
# 		a=[]
# 	gc.collect()
# 	return str(b)


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

  processed_features["distance_from_san_francisco"]=abs(california_housing_dataframe["latitude"]-38)

  # processed_features['latitude_%d_to_%d' % r]=select_and_transform_features(california_housing_dataframe['latitude'])
  # processed_features["binning_values"]=binning(california_housing_dataframe["latitude"])
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



# # 任务 1：构建良好的特征集

# # 如果只使用 2 个或 3 个特征，您可以获得的最佳效果是什么？

# # 相关矩阵展现了两两比较的相关性，既包括每个特征与目标特征之间的比较，也包括每个特征与其他特征之间的比较。

# # 在这里，相关性被定义为皮尔逊相关系数。您不必理解具体数学原理也可完成本练习。

# # 相关性值具有以下含义：

# #     -1.0：完全负相关
# #     0.0：不相关
# #     1.0：完全正相关


# # 定义相关性：
correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]

# print(correlation_dataframe.corr())

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
    
    # Convert pandas data into a dict of np arrays
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




def train_model(
    learning_rate,
    steps,
    batch_size,
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
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
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
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
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

  return linear_regressor
  # return ({'linear_regressor':linear_regressor})




# # 花 5 分钟时间来搜索一组效果良好的特征和训练参数。然后查看解决方案，看看我们选择了哪些参数。请谨记，不同的特征可能需要不同的学习参数。
# #
# # Your code here: add your features of choice as a list of quoted strings.
# #
# # minimal_features = ["latitude",
# # 	"longitude",
# # 	"housing_median_age"
# # ]


# # assert minimal_features, "You must select at least one feature!"

# # minimal_training_examples = training_examples[minimal_features]
# # minimal_validation_examples = validation_examples[minimal_features]

# # #
# # # Don't forget to adjust these parameters.
# # #
# # train_model(
# #     learning_rate=0.001,
# #     steps=500,
# #     batch_size=5,
# #     training_examples=minimal_training_examples,
# #     training_targets=training_targets,
# #     validation_examples=minimal_validation_examples,
# #     validation_targets=validation_targets)


# # 参考答案:
# minimal_features = [
#   "median_income",
#   "latitude",
# ]

# # minimal_training_examples = training_examples[minimal_features]
# # minimal_validation_examples = validation_examples[minimal_features]

# # _ = train_model(
# #     learning_rate=0.01,
# #     steps=500,
# #     batch_size=5,
# #     training_examples=minimal_training_examples,
# #     training_targets=training_targets,
# #     validation_examples=minimal_validation_examples,
# #     validation_targets=validation_targets)





# # 任务 2：更好地利用纬度

# # 绘制 latitude 与 median_house_value 的图形后，表明两者确实不存在线性关系。

# # 不过，有几个峰值与洛杉矶和旧金山大致相对应。

# # plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
# # plt.show()




# # 尝试创建一些能够更好地利用纬度的合成特征。

# # 例如，您可以创建某个特征，将 latitude 映射到值 |latitude - 38|，并将该特征命名为 distance_from_san_francisco。

# # 或者，您可以将该空间分成 10 个不同的分桶（例如 latitude_32_to_33、latitude_33_to_34 等）：
# # 如果 latitude 位于相应分桶范围内，则显示值 1.0；如果不在范围内，则显示值 0.0。

# # 使用相关矩阵来指导您构建合成特征；如果您发现效果还不错的合成特征，可以将其添加到您的模型中。

# # 您可以获得的最佳验证效果是什么？

# #
# # YOUR CODE HERE: Train on a new data set that includes synthetic features based on latitude.
# #

# # 使用distance_from_san_francisco(定义在56行)：
# minimal_features = [
#   "median_income",
#   "distance_from_san_francisco",
# ]

# minimal_training_examples = training_examples[minimal_features]
# minimal_validation_examples = validation_examples[minimal_features]

# _ = train_model(
#     learning_rate=0.01,
#     steps=500,
#     batch_size=5,
#     training_examples=minimal_training_examples,
#     training_targets=training_targets,
#     validation_examples=minimal_validation_examples,
#     validation_targets=validation_targets)

# # 我的结论：使用新特征值distance_from_san_francisco的效果不佳(RMSE为130.04)。

# 使用分箱：
# 参考答案：
# LATITUDE_RANGES = zip(xrange(32, 44), xrange(33, 45))				#xrange函数是python2的写法，在python3中用range来替代
# LATITUDE_RANGES = zip(range(33, 44), range(34, 45)) 
LATITUDE_RANGES = zip(range(32, 44), range(33, 45))					#range函数给的是一个区间，两个参数分别是区间的上下界(左闭右开)
#zip函数的用法，参考http://www.runoob.com/python/python-func-zip.html

def select_and_transform_features(source_df):
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  for r in LATITUDE_RANGES:					# 根据zip函数，每个r对应的就是一个长度为1的区间：(32,33),(33,34),(34,35)...
    selected_examples['latitude_%d_to_%d' % r] = source_df["latitude"].apply(
    # selected_examples["latitude_%dto_%d" % r] = source_df["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
  return selected_examples

selected_training_examples = select_and_transform_features(training_examples)
selected_validation_examples = select_and_transform_features(validation_examples)

_ = train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets)


# debug日志：
# 错误信息：ValueError: Feature latitude_32_to_33 is not in features dictionary.
# 以上错误应该是因为在processed_features中没有对新的分箱特征进行定义，但是如果要在前面定义的话，又会遇到在定义之前就调用变量的问题。目前暂时没有想到解决方法。
# 另外以上代码在Google的网页上是可以运行的，原因未知。。。