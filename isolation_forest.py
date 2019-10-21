import sklearn.ensemble
import sklearn.metrics
import numpy as np
from common import dataset

num_sevens = 0
seed = 0
d = dataset.Mnist_Fives_Small_Sevens_Dataset(num_fives=5000-num_sevens, num_sevens=num_sevens)


m = sklearn.ensemble.IsolationForest(n_jobs =-1, random_state=seed, contamination=num_sevens / 5000)
m.fit(d.get_train().reshape(-1, 784))
print('fitted')
d_test = dataset.Mnist_Fives_Small_Sevens_Dataset(num_fives=450, num_sevens=50)
# The lower the score the more abnormal
errors = m.score_samples(d_test.get_test().reshape(-1, 784))
errors = -errors

# Calculate array of metrics
anomaly_mask = (d_test.get_test_labels().flatten() == 7)
ap = sklearn.metrics.average_precision_score(anomaly_mask, errors)
auc = sklearn.metrics.roc_auc_score(anomaly_mask, errors)
# Calc P@10 metric, percent anomaly in top ten error scores
top_anomalies = np.argsort(errors)[::-1][:10]
p_at_10 = sklearn.metrics.accuracy_score(anomaly_mask[top_anomalies], np.ones(10))
print(ap, auc, p_at_10)
