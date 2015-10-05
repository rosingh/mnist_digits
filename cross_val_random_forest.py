import pandas as pd
import numpy as np
import csv as csv
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv', header=0)
train_data = train_df.values

print 'Cross Validating'
for i in range(0, 21):
	forest = RandomForestClassifier(n_estimators=10*i)
	fold_scores = cross_validation.cross_val_score(forest, train_data[0::, 1::], train_data[0::, 0], cv=5)
	mean_fold_score = fold_scores.mean()
	mean_fold_std = fold_scores.std()
	print("Accuracy: %0.5f (+/- %0.5f)" % (mean_fold_score, mean_fold_std *2))