# bankruptcy_prediction

The objective for the use case is to predict bankruptcy for various companies. It is a highly imbalanced with 3% of positive cases. In the notebooks, I have carried out analysis and developed the model. As a first step it contains a benchmark model using Linear Regression and then the final model chosen uses GBM type model framework called [Catboost](https://catboost.ai/en/docs/) wherein I used [Optuna](https://optuna.org/) for hyperparameter optimsation.

The key for such uses case is to inform the model of the imbalance and then tune. I have uses class weights to tune it along with several other hyperparameters. I could have used oversampling but chose not to do so as this mostly result in poor performance in the real world. The helper functions are avaiable in utils.

Enjoy the code!
