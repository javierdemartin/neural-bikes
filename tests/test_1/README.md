# Test 1

![Neural Network Shape](nn_shape.png)

> Neural Network shape stacking LSTM + Dense into a Sequential model

First test as a multivariate neural network providing as inputs

* Free bikes (what I want to predict)
* Weekday
* Month  (tried giving as input the date but the results weren't good)

## Results

|  Set  	| Training Set 	| Test Set 	| Epochs 	| Batch Size 	| RMSE         |
|:-----:	|:------------:	|:--------:	|:------:	|:----------:	| :----------: |
| 18033 	|      67%     	|    33%   	|   20   	|     80     	| 0.111        |

## Loss

![Loss Graph](loss.png)

## Next Steps

* Instead of giving as input `date`/`month` try `month` + `hour` to add more info and see if the predictions improve at all

