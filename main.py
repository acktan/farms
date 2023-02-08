import os
os.chdir(r'C:\Users\ckunt\OneDrive\Documents\Masters work\Extra\farms')
import json
import logging
from sklearn.ensemble import ExtraTreesRegressor
from src.loading import loading
from src.preprocessing import preprocessing
from src.train import train
from src.evaluation import evaluation
from src.inference import inference

logger = logging.getLogger(__name__)
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    loader = loading.DataLoader(config)
    df_train, future = loader.get_data()
    
    if config['train']['weather']:
        preprocess = preprocessing.Preprocessing(df_train)
        df_train = preprocess.get_hist_weather()
        preprocess_future = preprocessing.Preprocessing(future)
        future = preprocess_future.get_hist_weather()
    
    if config['train']['val']:
        preprocess = preprocessing.Preprocessing(df_train)
        X_train, X_test, y_train, y_test = preprocess.train_val_split()
    else:
        preprocess = preprocessing.Preprocessing(df_train)
        X_train, _, y_train, _ = preprocess.train_val_split(train_size=1)
        
    training = train.Train(X_train, y_train)
    pipe = training.train(model=ExtraTreesRegressor(random_state=42))
    
    #evaluation
    if config['train']['val']:
        evaluator = evaluation.Evaluator(config, X_train, pipe)
        evaluator.get_feature_importance()
        evaluator.compare_test_values(X_test, y_test)
        print('Test absolute mean error: {}'.format(evaluator.evaluate_model(X_test, y_test)))
    else: 
        futurePredict = inference.Inference(config, pipe, X_train, future)
        print('Future yields are {}'.format(futurePredict.predict_future()))
    
