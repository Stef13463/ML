import matplotlib.pyplot as plt
import seaborn as sns
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
        self.error = None

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

    def testRegression(self, weights, x_test=None):
        y_stern = []
        if x_test is None:
            x_test = self.raw_data["x_test"]
        for element in x_test:
            y_stern.append(self.f_x(weights, element))
        return y_stern

    def computeError(self, y_stern):
        y_test = self.raw_data["y_test"][0]
        self.error = []
        for i in range(len(y_star)):
            self.error.append(math.fabs(y_stern[i]-y_test[i]))
        mean_error = np.mean(self.error)
        return [self.error, mean_error]

    def plotError(self, error):
        sorted_error = np.sort(error)
        sorted_error = sorted_error[::-1]
        plt.plot(sorted_error, "b.")
        plt.show()


    def plotHeatMap(self):
        test_x = self.raw_data["x_test"]
        latitude = []
        longitude = []
        for i in range(len(test_x)):
            latitude.append(test_x[i][1])
            longitude.append(test_x[i][2])
        matrix = []
        row = []
        i = 0
        j = 0
        while latitude[i] == latitude[i-1]:
            row.append(self.error[i])
            i = i + 1
        matrix.append(row)

        err = self.error



if __name__ == "__main__":
    r = RidgeRegression(4)
    data = r.importData("data.mat")
    t_data = r.generateTrainingData(data[0], data[1])
    w = r.calculateWeights(t_data[0], t_data[1], 0.1)
    y_star = r.testRegression(w)
    err = r.computeError(y_star)
    # r.plotError(err[0])
    r.plotHeatMap()
    print(err)
