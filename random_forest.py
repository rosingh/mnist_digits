import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv', header=0)
train_data = train_df.values
test_df = pd.read_csv('test.csv', header=0)
test_data = test_df.values

forest = RandomForestClassifier(n_estimators=180)
forest.fit(train_data[0::, 1::], train_data[0::, 0])
output = forest.predict(test_data).astype(int)

predictions_file = open('forestdigits.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['ImageId', 'Label'])

open_file_object.writerows(zip(range(1, np.shape(test_data)[0] + 1), output))
predictions_file.close()
