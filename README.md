# Linear-Regression-of-Stock-Data-from-Scratch
Creating a linear regression model for Stock Market Data from scratch for an extremely low correlation for (Adj. Close vs Volumes) using just python lists and no other modules. 

Note: Numpy and Matplotlib are used but only for plotting and not for calculation/regression part.

Coming up with an alternate idea to increase correlation (Using a linear relation between Adj. Close price and Close price vs Volume)

Comparing with least square fitting method, finding cook's distance, and the outliers. 

Finding and removing the outliers and creating a new model. 

Plotting the points, the cook's distances and the final models, both of the linear relation vs volumes and adj. close prices vs volumes. 

Plotting both before and after finding outliers.

Looking at the final performances of both the original and outlier filtered models on both the training and the test dataset using RMSE errors.

New: Regress 2 shows the plot for the cost function with logarithmic cost, logarithmic slope (x) and bias (y). (Ignore the labels of x as it's values are wrong) Regress 3 does a normal linear regression but stops at cost function at around 2. This is because the drop is very shallow and for different alphas as well it skips the minima.


# Instructions to Download:

The regress.py file is just a file to test the regression model and is optional.

The stock_regression.py is the main python file. 

There are 2 datasets one for train and another for test. Download both of them.

Save them at the same destination folder.
