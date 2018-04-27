#!/usr/bin/python
# -*- coding: utf-8 -*-
# by:Adil



class SSK_SVC():
    # 此方法只适用于SNP数据或者普通字符串数据:
    # eg.:['ACGTAGCTAGCTAC', 'ACGTAGCGATCGATC',.....]
    # eg :['Are you ok', 'Yes, I'm ok']
    # clf为SVC分类器
    # kernel 为string核函数
    # feats_train为字符串数据转化为SVC输入的中间量，不需要使用者考虑
    def __init__(self):

        self.clf = None
        self.feats_train = None
        self.kernel = None
        self.isSNP = True

    # 训练数据
    # data 训练数据集,尽量使用list
    def fit(self, data, label, maxlen, decay, isSNP=True):
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from shogun import SNP, RAWBYTE
        from shogun import StringCharFeatures
        from shogun import SubsequenceStringKernel

        param = {'C': [1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 15, 20],
                 'probability': [True, False],
                 'decision_function_shape': ['ovo', 'ovr'],
                 'degree': [1, 2, 3, 4, 5]}
        svc = SVC(kernel='precomputed')
        self.clf = GridSearchCV(estimator=svc,
                                param_grid=param,
                                scoring='f1_macro',
                                n_jobs=6,
                                cv=5)
        try:
            if type(data) != type([]):
                data = data.tolist()
        except:
            print '请输入正确的data格式，只能是list或者np.array'

        # 将输入字符串转换成中间变量
        self.isSNP = isSNP
        if isSNP:
            self.feats_train = StringCharFeatures(data, SNP)
        else:
            self.feats_train = StringCharFeatures(data, RAWBYTE)
        # 获取核函数
        print '训练核函数,可能会花费较长时间'
        self.kernel = SubsequenceStringKernel(self.feats_train,
                                              self.feats_train,
                                              maxlen,
                                              decay)
        # 得到输入向量
        data = self.kernel.get_kernel_matrix()
        print '训练数据'
        self.clf.fit(data, label)
        print 'f1_macro的结果是:' + str(self.clf.best_score_)

    def predict(self, data):
        from shogun import SNP, RAWBYTE
        from shogun import StringCharFeatures
        if self.isSNP:
            feats_test = StringCharFeatures(data, SNP)
        else:
            feats_test = StringCharFeatures(data, RAWBYTE)
        # 将测试string数据转化为中间量
        self.kernel.init(self.feats_train, feats_test)
        feats_test = self.kernel.get_kernel_matrix()
        result = self.clf.predict(feats_test.T)
        print ' '.join(map(str, result))
        return result

    def score(self, data, label):
        from shogun import SNP, RAWBYTE
        from shogun import StringCharFeatures
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score
        if self.isSNP:
            feats_test = StringCharFeatures(data, SNP)
        else:
            feats_test = StringCharFeatures(data, RAWBYTE)
        # 将测试string数据转化为中间量
        self.kernel.init(self.feats_train, feats_test)
        feats_test = self.kernel.get_kernel_matrix()
        retult = self.clf.predict(feats_test.T)
        acc = accuracy_score(label, retult)
        f1 = f1_score(label, retult, average='macro')
        print '正确率是:' + str(acc), 'F1得分是:' + str(f1)
        return acc, f1
