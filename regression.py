import scipy.io as sio
import numpy as np
import math


class Regression:
    def __init__(self, train_step):
        self.train_step = train_step
        self.raw_data = None

    def importData(self, filename):
        self.raw_data = sio.loadmat(filename)
        x1 = self.raw_data["LatitudeScale"]
        x2 = self.raw_data["LongitudeScale"]
        y = self.raw_data["TempField"]

        raw_x = []
        raw_y = []

        for i in range(len(x1)):
            for j in range(len(x2)):
                raw_x.append([1, x1[i][0], x2[j][0]])
                raw_y.append(y[i][j][0])
        return [raw_x, raw_y]

    def generateTrainingData(self, x, y):
        training_data_x = []
        training_data_y = []
        step = 0
        for i in range(len(x)):
            if step % 4 == 0 and not math.isnan(y[i]):
                training_data_x.append(x[i])
                training_data_y.append(y[i])
            step = step + 1
        return [training_data_x, training_data_y]


class RidgeRegression(Regression):
    def __init__(self, train_step):
        super().__init__(train_step)

    def calculateWeights(self, training_x, training_y, parameter_lambda):
        A = np.array(training_x)
        y = np.array(training_y)
        A_T = np.transpose(A)
        A_T__times__A = np.matmul(A_T, A)
        dim = A_T__times__A.shape[0]
        I = np.identity(dim)
        bracket = A_T__times__A + parameter_lambda * I
        bracket_inverse = np.linalg.inv(bracket)
        k = np.matmul(bracket_inverse, A_T)
        return np.matmul(k, y)

    def f_x(self, weights, x):
        if len(weights) is not len(x):
            weights.insert(0, 1)
        weights = np.array(weights)
        x = np.array(x)
        return np.matmul(x, np.matrix.transpose(weights))

    def testRegression(self, weight, x_test):
        pass

    def computeError(self, x_test, y_test):
        pass

    def plotError(self):
        pass

    def plotHeatMap(self):
        pass


if __name__ == "__main__":
    r = RidgeRegression(4)
    data = r.importData("data.mat")
    t_data = r.generateTrainingData(data[0], data[1])
    w = r.calculateWeights(t_data[0], t_data[1], 0.1)
    k = r.f_x(w, [1, 37.87, 263.67])
    print(k)
