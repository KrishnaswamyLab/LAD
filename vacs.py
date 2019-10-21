import numpy as np
from sklearn import metrics
import models
import atongtf.dataset
import atongtf.util

atongtf.util.set_config(seed=0)

path = 'vacs_models'
d = atongtf.dataset.VACS_Dataset()
#model = models.Gradient_Penalty_Lipschitz_Network(path, data_shape=d.get_shape(), batch_size=128, conv=False, noise=0.2, beta_1=0)
#model.train(d.get_train(), 50000, sample_interval=4000)

#m = models.load_model(path + '/model_48000')
#predictions = m.predict(d.get_test())
#
#np.save(path + '/preds.npy', predictions)

preds = np.load(path + '/preds.npy')
print(preds)
preds = -preds

print(metrics.roc_auc_score(d.get_test_labels(), preds.squeeze()))

