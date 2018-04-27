from SSK_SVM import SSK_SVC
from sklearn.externals import joblib

train_data = joblib.load('data/train_data.pkl')
test_data = joblib.load('data/test_data.pkl')
train_label = joblib.load('data/train_label.pkl')

clf = SSK_SVC()
clf.fit(train_data, train_label, 5, 0.9)
print clf.predict(test_data)