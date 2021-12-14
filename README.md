# !pip install pandas==0.23.4 numpy==1.14.2
from sklearn.datasets import load_iris # 导入鸢尾花数据集
iris = load_iris() # 载入数据集
print('iris数据集特征')
print(iris.data[:10])

print('iris数据集标签')
print(iris.target[:10])
from sklearn import tree # 导入决策树包
clf = tree.DecisionTreeClassifier() #加载决策树模型
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
predictions[:10]

from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions))

from sklearn.datasets import load_iris # 导入鸢尾花数据集
from sklearn import tree # 导入决策树包
from sklearn.metrics import accuracy_score # 导入准确率评价指标
iris = load_iris() # 载入数据集
clf = tree.DecisionTreeClassifier(criterion = 'entropy') #更换criterion参数
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions))

from sklearn.datasets import load_iris # 导入鸢尾花数据集
from sklearn import tree # 导入决策树包
from sklearn.metrics import accuracy_score # 导入准确率评价指标
iris = load_iris() # 载入数据集
clf = tree.DecisionTreeClassifier(criterion = 'entropy') #更换criterion参数
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions))

from sklearn.datasets import load_iris # 导入鸢尾花数据集
from sklearn import tree # 导入决策树包
from sklearn.metrics import accuracy_score # 导入准确率评价指标
iris = load_iris() # 载入数据集
clf = tree.DecisionTreeClassifier(max_depth=2) #更换criterion参数
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions))

======================================================================================
#朴素贝叶斯预测
# !pip install pandas==0.23.4 numpy==1.14.2
from sklearn.datasets import load_iris # 导入鸢尾花数据集
iris = load_iris() # 载入数据集
print('iris数据集特征')
print(iris.data[:10])

print('iris数据集标签')
print(iris.target[:10])

from sklearn.naive_bayes import GaussianNB # 导入朴素贝叶斯包
clf = GaussianNB()

clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集

predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
predictions[:10]

from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('Accuracy:%s' % accuracy_score(iris.target[120:], predictions))
