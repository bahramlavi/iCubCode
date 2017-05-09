

import sys;
sys.path.insert(0,".../caffe/python/") # Set your caffe path here!
import caffe
import numpy as np
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

def load_Caffe(len):
    #set this on your own path.
    net = caffe.Net('.../caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                    '.../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)
    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('.../caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(len, 3, 227, 227)

    return net,transformer

def transforming_icub_data(day):
    basePath = ".../iCub/dataset/iCubWorld28/" #Set the iCub dataset path here.
    trainPath = basePath + 'train/'+ day+'/'
    testPath = basePath + 'test/'+ day+'/'
    listing_objs = ['cup', 'dishwashing-detergent', 'laundry-detergent', 'plate', 'soap', 'sponge', 'sprayer']
    train_lbl=[]
    test_lbl = []
    img_count_lbl = 1
    counter = 0
    len_train = 0
    len_test = 0

    for lst_sub_dir in listing_objs:
        dr = os.listdir(trainPath + lst_sub_dir + '/')
        for dr2 in dr:
            if dr2.startswith('.'): continue
            len_train += len(os.listdir(trainPath + lst_sub_dir + '/' + dr2))
            len_test += len(os.listdir(testPath + lst_sub_dir + '/' + dr2))

    net_train,transformer_train=load_Caffe(len_train)
    net_test,transformer_test=load_Caffe(len_test)

    print('****** Beging to transform iCub training data ******')
    for lst_sub_dir in listing_objs:
        dr = os.listdir(trainPath + lst_sub_dir + '/')
        for dr2 in dr:
            if dr2.startswith('.'): continue
            print ('Transofrming for object: ' + str(img_count_lbl))
            list_imgs_fileName = os.listdir(trainPath + lst_sub_dir + '/' + dr2)
            for img in list_imgs_fileName:
                im = caffe.io.load_image(trainPath + lst_sub_dir + '/' + dr2 + '/' + img)
                net_train.blobs['data'].data[counter, ...] = transformer_train.preprocess('data', im)
                train_lbl.append(img_count_lbl)
                counter += 1
            img_count_lbl += 1

    out=net_train.forward()
    trainSet = net_train.blobs['fc7'].data

    counter = 0
    img_count_lbl = 1
    print('****** Beging to transform iCub testing data ******')
    for lst_sub_dir in listing_objs:
        dr = os.listdir(trainPath + lst_sub_dir + '/')
        #if '.DS_Store' in dr: del dr['.DS_Store']
        for dr2 in dr:
            if dr2.startswith('.'): continue
            print ('Transforming for object: ' + str(img_count_lbl))
            list_imgs_fileName = os.listdir(testPath + lst_sub_dir + '/' + dr2)
            for img in list_imgs_fileName:
                im = caffe.io.load_image(testPath + lst_sub_dir + '/' + dr2 + '/' + img)
                net_test.blobs['data'].data[counter, ...] = transformer_test.preprocess('data', im)
                test_lbl.append(img_count_lbl)
                counter += 1
            img_count_lbl += 1

    net_test.forward()
    testSet=net_test.blobs['fc7'].data

    return trainSet,train_lbl,testSet,test_lbl

def SVM_fit(trainSet,train_lbl):
    clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', class_weight='auto')
    clf.fit(trainSet, train_lbl)
    return clf

def SVM_predecit(svm_model,testSet):
    test_lbl_pr = svm_model.predict(testSet)
    return test_lbl_pr #accuracy_score(test_lbl, test_lbl_pr)

def accuracy_boxplot_diff_subsets(trainSet_d,train_lbl_d,testSet_d,test_lbl_d):
    objs = np.arange(1, 29)
    obj_num = np.arange(2, 28, 2)
    accu_score = []
    total_accu_score = []
    for o in obj_num:
        print('# of objects %f', o)
        print('=======================================')
        for iter1 in range(1, 401):
            print(iter1)
            rnd_obj = np.unique(random.sample(objs, o))
            trainSet = []
            testSet = []
            train_lbl = []
            test_lbl = []
            a = []
            for r in rnd_obj:
                indx_objs = [i for i, x in enumerate(train_lbl_d) if x == r]
                trainSet = trainSet + (list(trainSet_d[i] for i in indx_objs))
                train_lbl = train_lbl + (list(train_lbl_d[i] for i in indx_objs))

                indx_objs = [i for i, x in enumerate(test_lbl_d) if x == r]
                test_lbl = test_lbl + (list(test_lbl_d[i] for i in indx_objs))
                testSet = testSet + (list(testSet_d[i] for i in indx_objs))

            clf=SVM_fit(trainSet,train_lbl)
            test_lbl_pr=SVM_predecit(clf,testSet)
            accu_score.append(accuracy_score(test_lbl, test_lbl_pr))

            if o == 28: break
        total_accu_score.append(accu_score)
        accu_score = []

    plt.boxplot(total_accu_score)
    plt.ylim([0.6, 1])
    plt.xlim([1.5, 29])

    confidences = [70, 80, 90, 95]
    colors = ('y', 'k', 'g', 'r')
    legends = ('70%', '80%', '90%', '95%')
    cnt_cl = 0
    for conf in confidences:
        conf_lev = []
        for i in np.arange(1, len(total_accu_score) - 1):
            print i
            conf_lev.append(conf_level(total_accu_score[i], confidence=conf))
        color = colors[cnt_cl]
        leg = legends[cnt_cl]
        conf_lev.append(total_accu_score[27][0])
        plt.plot(np.arange(2, 29), conf_lev, linestyle='--', linewidth=4, color=color, label=leg)
        cnt_cl += 1
    plt.legend(bbox_to_anchor=(0.9, 0.9), loc=1, borderaxespad=0.)
    plt.show()

def conf_level(data, confidence=90):
    pos=0.0
    data.sort(reverse=True)
    conf_value=0.0
    pos=confidence*400/100
    conf_value=data[pos]

    return conf_value


# Transforming all available training&testing data for all days.
trainSet_d1, train_lbl_d1,testSet_d1,test_lbl_d1 = transforming_icub_data('day1')
trainSet_d2, train_lbl_d2,testSet_d2,test_lbl_d2 = transforming_icub_data('day2')
trainSet_d3, train_lbl_d3,testSet_d3,test_lbl_d3 = transforming_icub_data('day3')
trainSet_d4, train_lbl_d4,testSet_d4,test_lbl_d4 = transforming_icub_data('day4')



# Computing the accuracy of predictor trained on Day1 and tested on Day1.
# One can laterally compute for other days, and compare with
# the results presented at Table 1 of the original paper.
print 'Accuracy of predictor trained on Day1 vs. Day1 = '+ accuracy_score(test_lbl_d1, SVM_predecit(SVM_fit(trainSet_d1,train_lbl_d1),testSet_d1)).__str__()


# Computing the box plot demonstrated on figure 4 of the original paper
# in which the accuracies measured for predictors trained on random subsets.
accuracy_boxplot_diff_subsets(trainSet_d4,train_lbl_d4,testSet_d4,test_lbl_d4)




