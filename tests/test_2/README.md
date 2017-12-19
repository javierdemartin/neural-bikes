# Test 2

## What did I do?

In the previous test I managed to get a RMSE value of 0.097 but an accuracy of 33%. Seems that the accuracy measurement done by Keras internally is not what I want, I think it compares the values and if they don't match 100% it is considered as an error.

For example, if the prediction for a station is  9.03 bikes and the actual value is 9 bikes it is considered by Keras as an error whereas if we looked as humans at that prediction we'd agree that is a correct value. So I'll try to round down every value to an integer to see if the accuracy increases. Only encode categorical variables, not continuous.

![Neural Network Shape](nn_shape.png)

> Neural Network shape stacking LSTM + Dense into a Sequential model

First test as a multivariate neural network providing as inputs

* Free bikes (what I want to predict)
* Weekday
* Month  (tried giving as input the date but the results weren't good)
* Hour

## Results

|  Set  	| Training Set 	| Test Set 	| Epochs 	| Batch Size 	| RMSE         |
|:-----:	|:------------:	|:--------:	|:------:	|:----------:	| :----------: |
| 18033 	|      67%     	|    33%   	|   20   	|     80     	| 0.097        |

## Accuracy

![Accuracy Graph](acc.png)

> This doesn't look good at all 🙃

## Loss

![Loss Graph](loss.png)

