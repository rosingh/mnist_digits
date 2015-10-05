import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm, cross_validation

train_df = pd.read_csv('train.csv', header=0)
train_data = train_df.values
test_df = pd.read_csv('test.csv', header=0)
test_data = test_df.values

sup = svm.LinearSVC(C=pow(10, -5))
sup.fit(train_data[0::, 1::], train_data[0::, 0])
output = sup.predict(test_data).astype(int)
print output

predictions_file = open('svmdigits.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['ImageId', 'Label'])
i = 1
for item in output:
	open_file_object.writerow([i, item])
	i = i + 1
predictions_file.close()
