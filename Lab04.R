### ASML Classification Lab 4 -- Deep Learning Intro ###
########################################################

# NOTE: This Lab requires at least 8GB RAM, so if you are using
#       Github Codespaces, make sure to launch on the 4-core server!

## Introduction
#
# Some of the software in this lab doesn't work well in notebooks and
# we want to know how to use ordinary script files anyway, since these
# are more commonly used for serious programming work.
#
# We will continue looking at the credit dataset from the last lab and
# do an analysis using the deep learning AI techniques that we learned
# in the last three lectures).
#
# First, load the dataset into R:

data("credit_data", package = "modeldata")

# We assume you remember what this data is like from previous labs!  If
# not, go back and remind yourself before continuing.
#
# We first split the data into train/validate/test.  We won't do cross
# validation here just because deep learning is the most computationally
# expensive method we've looked at so far and we don't have the time in
# the lab to wait for a full 5-fold cross validation to run.
# Do try yourself outside the lab.
library("rsample")
set.seed(212) # by setting the seed we know everyone will see the same results
# First get the training
credit_split <- initial_split(credit_data)
credit_train <- training(credit_split)
# Then further split the training into validate and test
credit_split2 <- initial_split(testing(credit_split), 0.5)
credit_validate <- training(credit_split2)
credit_test <- testing(credit_split2)

# The first thing we need to do is to transform the data into a form
# which works with deep learning.
#
# - Every feature must be numeric;
#
# - use one-hot coding for categorical data;
#
# - there must be no missing values;
#
# - each feature should be normalised to mean zero, standard deviation 1
# (or tranformed to the range [0,1])
#
# We use the recipes package to do this so you can see an alternative
# to the pipelines from MLR3 which gives you more manual control.
#
# One of the special things about the recipes package is that it
# enables computing the scaling factors on training data and to then use
# this on validation and testing without having to keep the training
# data around later on.  This is because it is *very* important that you
# do not independently scale and centre the validation and testing
# data!!  Just to repeat, because this is so important:
#
# ** Do not independently scale and centre the validation and testing data!! **
#
# The act of scaling and centring is part of the analysis
# pipeline and the mean and standard deviation estimated from training
# data are effectively parameters of the ultimate model.
library("recipes")

cake <- recipe(Status ~ ., data = credit_data) |>
  step_impute_mean(all_numeric()) |> # impute missings on numeric values with the mean
  step_center(all_numeric()) |> # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) |> # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) |> # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) |> # turn all factors into a one-hot coding
  prep(training = credit_train) # learn all the parameters of preprocessing on the training data

credit_train_final <- bake(cake, new_data = credit_train) # apply preprocessing to training data
credit_validate_final <- bake(cake, new_data = credit_validate) # apply preprocessing to validation data
credit_test_final <- bake(cake, new_data = credit_test) # apply preprocessing to testing data

# Have a look at this data to see what has been done



## Keras
#
# Although MLR3 does have support for fitting deep learning models as
# part of its pipeline, for the first time you use them we'd like to
# have full control (and often you want more fine detail control when
# fitting deep learning than is needed with classical ML methods).
#
# Therefore we'll directly interface to Keras, a deep learning interface
# developed by François Chollet from Google which builds Tensorflow models.

library("keras")

# The lab server is already setup with Keras, but if you are running
# this on your own machine then you may need to install it with the
# command install_keras().  Please do NOT run that command on the lab
# server as it will download gigabytes of data for every user
# unnecessarily!

# We have one more data preparation step to perform, because Keras
# expects to receive the data in matrix form and wants the features and
# responses separately

credit_train_x <- credit_train_final |>
  select(-starts_with("Status_")) |>
  as.matrix()
credit_train_y <- credit_train_final |>
  select(Status_bad) |>
  as.matrix()

credit_validate_x <- credit_validate_final |>
  select(-starts_with("Status_")) |>
  as.matrix()
credit_validate_y <- credit_validate_final |>
  select(Status_bad) |>
  as.matrix()

credit_test_x <- credit_test_final |>
  select(-starts_with("Status_")) |>
  as.matrix()
credit_test_y <- credit_test_final |>
  select(Status_bad) |>
  as.matrix()



# We can now start to construct our deep neural network architecture
# We make a neural network with two hidden layers, 32 neurons in the
# first, 32 in second and an output to a binary classification
deep.net <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(credit_train_x))) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

# This must then be "compiled".  See lectures on the optimiser.
deep.net |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Finally, fit the neural network!  We provide the training data, and
# also a list of validation data.  We can use this to monitor for
# overfitting. See lectures regarding mini batches
deep.net |> fit(
  credit_train_x, credit_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(credit_validate_x, credit_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net |> predict(credit_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net |> predict(credit_test_x) |> `>`(0.5) |> as.integer()

# Confusion matrix/accuracy/AUC metrics
# (recall, in Lab03 we got accuracy ~0.80 and AUC ~0.84 from the super learner,
# and around accuracy ~0.76 and AUC ~0.74 from best other models)
table(pred_test_res, credit_test_y)
yardstick::accuracy_vec(as.factor(credit_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(credit_test_y, levels = c("1","0")),
                       c(pred_test_prob))



# Ok, we didn't really go that deep?  Try deeper??
deep.net <- keras_model_sequential() |>
  layer_dense(units = 128, activation = "relu",
              input_shape = c(ncol(credit_train_x))) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

deep.net |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net |> fit(
  credit_train_x, credit_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(credit_validate_x, credit_validate_y),
)



# Ok, that looks like a bad idea!  Massive overfitting.

# We learned in lectures that we have methods that can combat this and still
# allow fitting very deep neural networks:
deep.net <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(credit_train_x))) |>
  layer_batch_normalization() |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_batch_normalization() |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_batch_normalization() |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_batch_normalization() |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

deep.net |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net |> fit(
  credit_train_x, credit_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(credit_validate_x, credit_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net |> predict(credit_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net |> predict(credit_test_x) |> `>`(0.5) |> as.integer()

# Confusion matrix/accuracy/AUC metrics
# (recall, in Lab03 we got accuracy ~0.80 and AUC ~0.84 from the super learner,
# and around accuracy ~0.76 and AUC ~0.74 from best other models)
table(pred_test_res, credit_test_y)
yardstick::accuracy_vec(as.factor(credit_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(credit_test_y, levels = c("1","0")),
                       c(pred_test_prob))

# In practice you would want to now do a careful exercise in cross-validation
# searching over architectures and regularisation schemes, also investigating
# the optimiser options.  See lectures for more discussion of this.
#
# For now, it is more important we see how to build a more complex deep neural
# network example.
#
# We revisit the MNIST handwriting example from Lab 2.  Download (if needed)
# and read the data.
download.file("http://www.louisaslett.com/Courses/MISCADA/mnist.fst", "mnist.fst")
mnist <- fst::read.fst("mnist.fst")

# Recall this code allows you to look at any image
library("tidyverse")

plotimages <- function(i) {
  imgs <- mnist |>
    slice(i) |>
    mutate(image.num = i) |>
    pivot_longer(x0.y27:x27.y0,
                 names_to = c("x", "y"),
                 names_pattern = "x([0-9]{1,2}).y([0-9]{1,2})",
                 names_transform = list(x = as.integer, y = as.integer),
                 values_to = "greyscale")

  ggplot(imgs, aes(x = x, y = y, fill = greyscale)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black") +
    facet_wrap(~ image.num + response, labeller = label_both)
}

plotimages(21:24)

# We are going to fit a simple convolutional neural network to this data.
# We need to put this in tensor form to work with Keras.  So this is now a
# "three-dimensional matrix" with indices [image,x pixel,y pixel]
mnist_tensor <- array(c(as.matrix(mnist[,-1][784:1])), dim = c(60000,28,28))
mnist_tensor <- mnist_tensor[,28:1,] # this corrects the mirrored x-axis from the original import

# We can use a simple base R to plot any image we want.  Change the 5 below to
# any other value from 1 to 60000 to view it
image(1:28, 1:28, mnist_tensor[5,,])

# In fact, Keras expects colour channels, so for a greyscale image we must add
# an 'empty' dimension to make it a 4D tensor
dim(mnist_tensor) <- c(60000,28,28,1)


# The values are already mixed, so we take the first 40000 as training, and
# 10000 each of validation and testing
#
# We divide by 255 to get in the scale [0,1]
# We use class.ind rather than a whole recipe as we only need to create one-hot
# coding and no other pre processing pipeline
# The drop=FALSE prevents the 'empty' dimension being dropped
mnist_train_x <- mnist_tensor[1:40000,,,,drop=FALSE]/255
mnist_train_y <- nnet::class.ind(mnist[1:40000,1])

mnist_val_x <- mnist_tensor[40001:50000,,,,drop=FALSE]/255
mnist_val_y <- nnet::class.ind(mnist[40001:50000,1])

mnist_test_x <- mnist_tensor[50001:60000,,,,drop=FALSE]/255
mnist_test_y <- nnet::class.ind(mnist[50001:60000,1])


# Now we create a simple convolutional neural network
deep.net <- keras_model_sequential() |>
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(28,28,1)) |>
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_dropout(rate = 0.25) |>
  layer_flatten() |>
  layer_dense(units = 64, activation = 'relu') |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 10, activation = 'softmax')

deep.net |> compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# We should train for more epochs but don't want lab machine to go too slow!
# Try improving on this in your own time
deep.net |> fit(
  mnist_train_x, mnist_train_y,
  batch_size = 32,
  epochs = 5,
  validation_data = list(mnist_val_x, mnist_val_y)
)

pred <- deep.net |> predict(mnist_test_x) |> k_argmax() |> as.vector()
# Confusion matrix/accuracy
table(pred, max.col(mnist_test_y)-1)
yardstick::accuracy_vec(as.factor(max.col(mnist_test_y)-1),
                        as.factor(pred))

# Have a look at an image it got wrong ... change the i variable to see others
wrong <- which(pred != max.col(mnist_test_y)-1)
i <- 1
image(mnist_test_x[wrong[i],,,1],
      main = glue::glue("Truth: {(max.col(mnist_test_y)-1)[wrong[i]]}, Predicted: {pred[wrong[i]]}"))

# How would a random forest have done?
rf <- ranger::ranger(as.factor(response) ~ ., mnist[1:40000,])
predrf <- predict(rf, mnist[50001:60000,])
# Confusion matrix/accuracy
table(predrf$predictions, max.col(mnist_test_y)-1)
yardstick::accuracy_vec(as.factor(max.col(mnist_test_y)-1),
                        predrf$predictions)
