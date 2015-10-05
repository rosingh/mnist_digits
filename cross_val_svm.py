import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm, cross_validation

train_df = pd.read_csv('train.csv', header=0)
train_data = train_df.values

print 'Cross Validating'

print 'rbf kernel'

#for i in range(-5, 5):
#	sup = svm.SVC(C=pow(10, i))
#	fold_scores = cross_validation.cross_val_score(sup, train_data[0::, 1::], train_data[0::, 0], cv=3)
#	mean_fold_score = fold_scores.mean()
#	mean_fold_std = fold_scores.std()
#	print 'Accuracy: %0.5f (+/- %0.5f) % (mean_fold_score, mean_fold_std *2)'
print('Linear kernel')
for i in range(-5, 5):
	sup = svm.LinearSVC(C=pow(10, i))
	fold_scores = cross_validation.cross_val_score(sup, train_data[0::, 1::], train_data[0::, 0], cv=3)
	mean_fold_score = fold_scores.mean()
	mean_fold_std = fold_scores.std()
	print("Accuracy: %0.5f (+/- %0.5f)" % (mean_fold_score, mean_fold_std *2))