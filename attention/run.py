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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(0)

def load_dataset(task):
    X_train_tid, X_train, y_train, word_embeddings = pickle.load(open("dataset/"+task+"/train_new.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open("dataset/"+task+"/test_new.pkl", 'rb'))
    
    return X_train[0], y_train, \
           X_test[0], y_test, word_embeddings

def train_and_test(task):

    class TransformerBlock(nn.Module):

        def __init__(self, input_size=300, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
            super(TransformerBlock, self).__init__()
            embedding_weights = word_embeddings
            V, D = embedding_weights.shape
            self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))


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
            self.linear1 = nn.Linear(input_size, input_size)
            self.linear2 = nn.Linear(input_size, input_size)

            self.dropout = nn.Dropout(attn_dropout)
            self.fc2 = nn.Linear(300, 4)
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


        def forward(self, X_text):
            '''
            :param Q: (batch_size, max_q_words, input_size)
            :param K: (batch_size, max_k_words, input_size)
            :param V: (batch_size, max_v_words, input_size)
            :return:  output: (batch_size, max_q_words, input_size)  same size as Q
            '''

            X_text = self.word_embedding(X_text) # (N*C, W, D)
            V_att = self.multi_head_attention(X_text, X_text, X_text)

            if self.is_layer_norm:
                X = self.layer_morm(X_text + V_att)  # (batch_size, max_r_words, embedding_dim)
                output = self.layer_morm(self.FFN(X) + X)
            else:
                X = X_text + V_att
                output = self.FFN(X) + X
         
            output =torch.mean(output, dim=1)
            #print(output.shape)
            output = self.fc2(output)

            #print(output.shape)
            return output

    X_train, y_train, \
    X_test, y_test, word_embeddings = load_dataset(task)
    model= TransformerBlock()
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    epoch_num=30
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        print("\nEpoch ", epoch+1, "/", epoch_num)
        model.train()
        avg_loss = 0
        avg_acc = 0
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_text, batch_y = (item.cpu() for item in data)
            #print(batch_x_text)
            optimizer.zero_grad()
            logit = model(batch_x_text)
            loss = loss_func(logit, batch_y)
            loss.backward()
            optimizer.step()

            corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
            accuracy = 100*corrects/len(batch_y)

            avg_loss += loss.item()
            avg_acc += accuracy

        print("test part")
        y_pred = predict(model, X_test)
        print(classification_report(y_test, y_pred, digits=3))
        print(accuracy_score(y_test, y_pred))

def evaluate(model, X_dev, y_dev):
    y_pred = predict(model, X_dev)
    acc = accuracy_score(y_dev, y_pred)
    #print(classification_report(y_dev, y_pred, digits=5))
    print(acc)


def predict(model, X_test):

    model.eval()
    y_pred = []
    X_test = torch.LongTensor(X_test)

    dataset = TensorDataset(X_test)
    dataloader = DataLoader(dataset, batch_size=50)

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            logits = model(data[0])
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
    return y_pred




if __name__ == '__main__':
    task = 'twitter16_new'
    # task = 'twitter16'
    print("task: ", task)
    train_and_test(task)

