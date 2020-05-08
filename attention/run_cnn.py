import os
import pickle
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import itertools
from collections import Counter
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_dataset(task):
    X_train_tid, X_train, y_train, word_embeddings = pickle.load(open("dataset/"+task+"/train_new.pkl", 'rb'))
    #X_dev_tid, X_dev, y_dev = pickle.load(open("dataset/"+task+"/dev_new.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open("dataset/"+task+"/test_new.pkl", 'rb'))
    
    return X_train, y_train, \
           X_test, y_test, word_embeddings


def modifyFormat(X_train):

    X_follow = [int(x) if x != 'UNK' else 0 for x in X_train[1]]
    X_friend = [int(x) if x != 'UNK' else 0 for x in X_train[2]]
    X_ratio = [float(x) if x != 'UNK' else 0 for x in X_train[3]]
    X_verified = [1 if x != 'False' else 0 for x in X_train[4]]
    X_registration = [int(x) if x != 'UNK' else 0 for x in X_train[5]]
    X_tweets = [int(x) if x != 'UNK' else 0 for x in X_train[6]]

    return [X_follow, X_friend, X_ratio, X_verified, X_registration, X_tweets]



def prepareInput(X_train, follow_idx, friend_idx, ratio_idx, verified_idx, registration_idx, tweets_idx):
    X_follow = [[follow_idx[x]] for x in X_train[0]]
    X_follow = torch.LongTensor(X_follow)
    X_friend = [[friend_idx[x]] for x in X_train[1]]
    X_friend = torch.LongTensor(X_friend)
    X_ratio = [[ratio_idx[x]] for x in X_train[2]]
    X_ratio = torch.LongTensor(X_ratio)
    X_verified = [[verified_idx[x]] for x in X_train[3]]
    X_verified = torch.LongTensor(X_verified)
    X_registration = [[registration_idx[x]] for x in X_train[4]]
    X_registration = torch.LongTensor(X_registration)
    X_tweets = [[tweets_idx[x]] for x in X_train[5]]
    X_tweets = torch.LongTensor(X_tweets)

    return X_follow, X_friend, X_ratio, X_verified, X_registration, X_tweets




def buildEmbeddingAndIdx(sentences):
    
    list_set = set(sentences)
    valuetoidx = {} 
    for idx, elem in enumerate(list_set):
        valuetoidx[elem] = idx

    embedding = np.zeros(shape=(len(valuetoidx), 50), dtype='float32')

    return embedding, valuetoidx


def train_and_test(task):

    class TransformerBlock(nn.Module):

        def __init__(self, input_size=300, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
            super(TransformerBlock, self).__init__()
            embedding_weights = word_embeddings
            V, D = embedding_weights.shape

            self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))

            embedding_weights = follow_embeddings
            V, D = embedding_weights.shape
            print
            self.follow_embeddings = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))

            embedding_weights = friend_embeddings
            V, D = embedding_weights.shape
            self.friend_embeddings = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))

            embedding_weights = ratio_embeddings
            V, D = embedding_weights.shape
            self.ratio_embeddings = nn.Embedding(V, D , padding_idx=0, _weight=torch.from_numpy(embedding_weights))

            embedding_weights = verified_embeddings
            V, D = embedding_weights.shape
            self.verified_embeddings = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))

            embedding_weights = registration_embeddings
            V, D = embedding_weights.shape
            self.registration_embeddings = nn.Embedding(V, D, padding_idx=0,_weight=torch.from_numpy(embedding_weights))

            embedding_weights = tweets_embeddings
            V, D = embedding_weights.shape
            self.tweets_embeddings = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))


            self.n_heads = n_heads
            self.d_k = d_k if d_k is not None else input_size
            self.d_v = d_v if d_v is not None else input_size

            self.is_layer_norm = is_layer_norm
            if is_layer_norm:
                self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

            self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
            self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
            self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

            self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))

            self.filter_num_width = [(25, 1), (50, 2)]
            self.convolutions = []
            for out_channel, filter_width in self.filter_num_width:
                self.convolutions.append(
                    nn.Conv2d(
                        1,           # in_channel
                        out_channel, # out_channel
                        kernel_size=(50, filter_width), # (height, width)
                        bias=True
                    )
            )
            self.linear1 = nn.Linear(input_size, input_size)
            self.linear2 = nn.Linear(input_size, input_size)

            self.dropout = nn.Dropout(attn_dropout)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(125, 50)
            self.fc2 = nn.Linear(50, 4)
            self.__init_weights__()
            print(self)

        def __init_weights__(self):
            init.xavier_normal_(self.W_q)
            init.xavier_normal_(self.W_k)
            init.xavier_normal_(self.W_v)
            init.xavier_normal_(self.W_o)

            init.xavier_normal_(self.linear1.weight)
            init.xavier_normal_(self.linear2.weight)

        def FFN(self, X):
            output = self.linear2(F.relu(self.linear1(X)))
            output = self.dropout(output)
            return output

        def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
            '''
            :param Q: (*, max_q_words, n_heads, input_size)
            :param K: (*, max_k_words, n_heads, input_size)
            :param V: (*, max_v_words, n_heads, input_size)
            :param episilon:
            :return:
            '''
            temperature = self.d_k ** 0.5
            Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
            Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
            Q_K_score = self.dropout(Q_K_score)

            V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
            return V_att


        def multi_head_attention(self, Q, K, V):
            bsz, q_len, _ = Q.size()
            bsz, k_len, _ = K.size()
            bsz, v_len, _ = V.size()

            Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
            K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
            V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

            Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
            K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
            V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

            V_att = self.scaled_dot_product_attention(Q_, K_, V_)
            V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
            V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

            output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
            return output

        def conv_layers(self, x):
            chosen_list = list()
            for conv in self.convolutions:
                feature_map = torch.tanh(conv(x))
                chosen = torch.max(feature_map, 3)[0]            
                chosen = chosen.squeeze()
                chosen_list.append(chosen)
            return torch.cat(chosen_list, 1)

        def forward(self, X_train):
            '''
            :param Q: (batch_size, max_q_words, input_size)
            :param K: (batch_size, max_k_words, input_size)
            :param V: (batch_size, max_v_words, input_size)
            :return:  output: (batch_size, max_q_words, input_size)  same size as Q
            '''
            
            X_text = self.word_embedding(X_train[0]) # (N*C, W, D)
           
            X_follow = self.follow_embeddings(X_train[1]) 
            X_friend = self.friend_embeddings(X_train[2]) 
            X_ratio = self.ratio_embeddings(X_train[3]) 
            X_verified = self.verified_embeddings(X_train[4]) 
            X_registration = self.registration_embeddings(X_train[5]) 
            X_tweets = self.tweets_embeddings(X_train[6]) 
            
            V_att = self.multi_head_attention(X_text, X_text, X_text)

            if self.is_layer_norm:
                X = self.layer_morm(X_text + V_att)  # (batch_size, max_r_words, embedding_dim)
                output = self.layer_morm(self.FFN(X) + X)
            else:
                X = X_text + V_att
                output = self.FFN(X) + X
        
            #output =torch.mean(output, dim=1, keepdim=True)
            X_text = output.permute(0, 2, 1)
            information = torch.cat((X_follow, X_friend,X_ratio,X_verified, X_registration,X_tweets), dim=1)
            information = torch.transpose(information.view(information.size()[0], 1, information.size()[1], -1), 2, 3)
            conv_feature = self.conv_layers(information)
            X_text = torch.mean(X_text, dim=1)
            feature = torch.cat((conv_feature, X_text), dim=1)
            
            #print(feature.shape)
            a1 = self.relu(self.fc1(feature))
            d1 = self.dropout(a1)

            output = self.fc2(d1)

            #print(output.shape)
            return output

    X_train, y_train, \
    X_test, y_test, word_embeddings = load_dataset(task)

    X_text = torch.LongTensor(X_train[0])
    y_train = torch.LongTensor(y_train)

    X_test_text = torch.LongTensor(X_test[0])
    y_test = torch.LongTensor(y_test)

    X_train = modifyFormat(X_train)
    X_test = modifyFormat(X_test)
    
    follow_embeddings, follow_idx = buildEmbeddingAndIdx(X_train[0]+X_test[0])
    friend_embeddings, friend_idx = buildEmbeddingAndIdx(X_train[1]+X_test[1])
    ratio_embeddings, ratio_idx = buildEmbeddingAndIdx(X_train[2]+X_test[2])
    verified_embeddings, verified_idx = buildEmbeddingAndIdx(X_train[3]+X_test[3])
    registration_embeddings, registration_idx = buildEmbeddingAndIdx(X_train[4]+X_test[4])
    tweets_embeddings, tweets_idx = buildEmbeddingAndIdx(X_train[5]+X_test[5])

    X_follow, X_friend, X_ratio, X_verified, X_registration, X_tweets = prepareInput(X_train, follow_idx, friend_idx, ratio_idx, verified_idx, registration_idx, tweets_idx)
    X_test_follow, X_test_friend, X_test_ratio, X_test_verified, X_test_registration, X_test_tweets = prepareInput(X_test, follow_idx, friend_idx, ratio_idx, verified_idx, registration_idx, tweets_idx)


    model= TransformerBlock()
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(X_text, X_follow, X_friend, X_ratio, X_verified, X_registration, X_tweets, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    epochs_num = 10
    for epoch in range(epochs_num):
        print("\nEpoch ", epoch+1, "/", epochs_num)
        model.train()
        avg_loss = 0
        avg_acc = 0
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_text, batch_x_follow, batch_x_friend, batch_x_ratio, batch_x_verified, batch_x_registration, batch_x_tweets, batch_y = (item.cpu() for item in data)
            

            logit = model([batch_x_text, batch_x_follow, batch_x_friend, batch_x_ratio, batch_x_verified, batch_x_registration, batch_x_tweets])
            loss = loss_func(logit, batch_y)
            loss.backward()
            optimizer.step()

            corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
            accuracy = 100*corrects/len(batch_y)

            #print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(i, loss.item(), accuracy, corrects, batch_y.size(0)))
            if i > 0 and i % 100 == 0:
                evaluate(model, X_dev_text, X_dev_follow, X_dev_friend, X_dev_ratio, X_dev_verified, X_dev_registration, X_dev_tweets, y_dev)
                model.train()

            avg_loss += loss.item()
            avg_acc += accuracy


        #evaluate(model, X_dev, y_dev)

        print("test part")
        y_pred = predict(model, X_test_text, X_test_follow, X_test_friend, X_test_ratio, X_test_verified, X_test_registration, X_test_tweets)
        
        print(accuracy_score(y_test, y_pred))
'''
config = {
    'reg':0,
    'batch_size':64,
    'dropout':0.5,
    'maxlen':50,
    'epochs':20,
    'num_classes':4,
    'target_names':['NR', 'FR', 'UR', 'TR']
}
'''

def evaluate(model, X_dev, X_dev_follow, X_dev_friend, X_dev_ratio, X_dev_verified, X_dev_registration, X_dev_tweets, y_dev):
    y_pred = predict(model, X_dev, X_dev_follow, X_dev_friend, X_dev_ratio, X_dev_verified, X_dev_registration, X_dev_tweets)
    acc = accuracy_score(y_dev, y_pred)
    #print(classification_report(y_dev, y_pred, digits=5))
    print(acc)


def predict(model, X_test, X_test_follow, X_test_friend, X_test_ratio, X_test_verified, X_test_registration, X_test_tweets):

    model.eval()
    y_pred = []
    X_test = torch.LongTensor(X_test)
    dataset = TensorDataset(X_test, X_test_follow, X_test_friend, X_test_ratio, X_test_verified, X_test_registration, X_test_tweets)
    dataloader = DataLoader(dataset, batch_size=50)

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            batch_x_text, batch_x_follow, batch_x_friend, batch_x_ratio, batch_x_verified, batch_x_registration, batch_x_tweets = (item.cpu() for item in data)
            logits = model([batch_x_text, batch_x_follow, batch_x_friend, batch_x_ratio, batch_x_verified, batch_x_registration, batch_x_tweets])
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
    return y_pred




if __name__ == '__main__':
    task = 'twitter15'
    # task = 'twitter16'
    print("task: ", task+"_new")
    train_and_test(task+"_new")

