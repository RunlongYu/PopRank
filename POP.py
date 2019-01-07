# Implement PopRank.
# PopRank is a basic algorithm in collaborative filtering,
# which ranks the items according to their popularity.
# @author Runlong Yu, Han Wu

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import scores

class POP:
    user_count = 943
    item_count = 1682
    popularity = 20
    train_data_path = 'train.txt'
    test_data_path = 'test.txt'
    size_u_i = user_count * item_count
    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train, popularity):
        pop_item = [0 for _ in range(popularity)]
        pop_index = [0 for _ in range(popularity)]
        for i in range(self.item_count):
            p = 0
            for j in range(self.user_count):
                if i in user_ratings_train[j]:
                    p = p + 1
            if p > min(pop_index):
                pop_index[pop_index.index(min(pop_index))] = p
                pop_item[pop_index.index(min(pop_index))] = i
        predict_matrix = np.zeros((self.user_count, self.item_count))
        for i in range(self.user_count):
            for j in range(self.item_count):
                if j in pop_item:
                    predict_matrix[i][j] = 1
        return predict_matrix

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        # training
        predict_matrix = self.train(user_ratings_train, self.popularity)
        # prediction
        self.predict_ = predict_matrix.reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict

if __name__ == '__main__':
    pop = POP()
    pop.main()
