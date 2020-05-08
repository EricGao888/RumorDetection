import pruning.utility as utility
import pruning.TD_RvNN as TD_RvNN
import torch
import time
import datetime
import numpy as np
import sys
import torch.optim as optim

Nepoch = 20
lr = 0.001 #learning rate
use_cf_features = True
output_path = '../output/fold%s/' % utility.fold

if __name__ == "__main__":
    #load data
    tree_train, word_train, index_train, parent_num_train, cf_features_train, y_train, \
    tree_test, word_test, index_test, parent_num_test, cf_features_test, y_test = utility.loadData()
    cf_features_train = np.array(cf_features_train) / np.max(np.array(cf_features_train), axis=0)
    cf_features_test = np.array(cf_features_test) / np.max(np.array(cf_features_test), axis=0)
    # print(cf_features_train[0])
    # print(cf_features_test[0])
    #initialize model
    model = TD_RvNN.RvNN(use_cf_features=use_cf_features)
    #training and testing
    losses = []
    accuracies = []
    f1s = []
    for epoch in range(Nepoch):
        optimizer = optim.Adam(model.params, lr)
        # optimizer = optim.SGD(model.params, lr, momentum=0.9)
        for i in range(len(y_train)):
            model.zeroGrad()
            pred_y = model.compute_tree(word_train[i], index_train[i], parent_num_train[i], tree_train[i],
                                        cf_features_train[i])
            loss = torch.sum((torch.sub(torch.FloatTensor(y_train[i]),pred_y))**2)
            loss.backward()
            optimizer.step()
            losses.append(np.round(loss.detach(),2))
        print("epoch: {}, loss: {}".format(epoch + 1, np.mean(losses)))
        sys.stdout.flush()
        ## calculate loss and evaluate
        sys.stdout.flush()
        prediction = []
        for j in range(len(y_test)):
           pred_y = model.compute_tree(word_test[j], index_test[j], parent_num_test[j], tree_test[j], cf_features_test[j])
           prediction.append(pred_y)
        accuracy, micro_f1 = utility.evaluation_4class(prediction, y_test)
        accuracies.append(accuracy)
        f1s.append(micro_f1)
        print("Accuracy: %f, Micro-F1: %f" % (accuracy, micro_f1))
        # print('results: {}'.format(result))
        sys.stdout.flush()
    # time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # f = open(output_path + time + ".log", "w")
    # f.write("#with cf features, no pruning\n")
    # for i in range(Nepoch):
        # f.write("%d %f %f\n" % (i + 1, accuracies[i], f1s[i]))
    # f.close()