import csv
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import cm


class ContinuousDistribution(ABC):
    def __init__(self, dimension):
        if dimension <= 0:
            raise Exception("DimensionError")
        self.dimension = dimension
        self.dataByVariable = []

    def importCSV(self, CSVFilename):
        self.dataByVariable = [[] for _ in range(self.dimension)]
        with open(CSVFilename, "r") as myFile:
            csvFile = csv.reader(myFile)
            for row in csvFile:
                if len(row) != self.dimension:
                    raise Exception("DimensionError")
                for i in range(0, self.dimension):
                    self.dataByVariable[i].append(float(row[i]))

    def exportCSV(self, filename):
        with open(filename, "w", newline="") as myFile:
            csvFile = csv.writer(myFile)
            length = len(self.dataByVariable[0])
            for i in range(0, length):
                row = []
                for variable in self.dataByVariable:
                    row.append(str(variable[i]))
                csvFile.writerow(row)

    def getMean(self):
        meanVector = []
        for variable in self.dataByVariable:
            mean = 0
            mySum = 0
            for value in variable:
                mySum = mySum + float(value)
            mean = mySum / len(variable)
            meanVector.append(mean)
        return meanVector

    def getStandardDeviation(self):
        sd_vector = []
        for variable in self.dataByVariable:
            array = np.array(variable)
            sd = np.std(variable)
            sd_vector.append(sd)
        return sd_vector

    @abstractmethod
    def generateData(self, **kwargs):
        pass

    @abstractmethod
    def plotData(self):
        pass


# ---------------------------------------NormalDistribution-------------------------------------------
# ----------------------------------------------------------------------------------------------------
class GaussDistribution(ContinuousDistribution):

    def __init__(self, dimension):
        super().__init__(dimension)

    def generateData(self, mu, sigma, data_points):
        self.dataByVariable.clear()
        for i in range(self.dimension):
            x = np.random.normal(mu[i], sigma[i], data_points)
            self.dataByVariable.append(x)

    def plotData(self, plotType):
        if self.dimension == 1:
            self.__plot1D(plotType)
        elif self.dimension == 2:
            self.__plot2D(plotType)
        else:
            raise Exception("To many Dimension for plotting")

    def __plot1D(self, plotType):
        if plotType == "raw_data":
            plt.plot(self.dataByVariable[0], "go")
            plt.ylabel("Value")
            plt.xlabel("Sample")
            plt.show()

        if plotType == "distribution":
            my = self.getMean()[0]
            sigma = self.getStandardDeviation()[0]
            x = np.linspace(my - 3 * sigma, my + 3 * sigma, 1000)
            plt.plot(x, stats.norm.pdf(x, my, sigma))
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

    def __plot2D(self, plotType):
        if plotType == "raw_data":
            plt.plot(self.dataByVariable[0], self.dataByVariable[1], "go")
            plt.ylabel("x2")
            plt.xlabel("x1")
            plt.show()
        elif plotType == "distribution":
            my = self.getMean()
            covMatrix = np.cov(self.dataByVariable)
            x = np.linspace(my[0] - 3 * covMatrix[0, 0], my[0] + 3 * covMatrix[0, 0], 1000)
            y = np.linspace(my[1] - 3 * covMatrix[1, 1], my[1] + 3 * covMatrix[1, 1], 1000)
            X, Y = np.meshgrid(x, y)
            rv = stats.multivariate_normal(my, covMatrix)
            pos = np.dstack((X, Y))
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X, Y, rv.pdf(pos), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.ylabel("x2")
            plt.xlabel("x1")
            plt.show()


# ---------------------------------------BetaDistribution------------------------------------------
# -------------------------------------------------------------------------------------------------

class BetaDistribution(ContinuousDistribution):
    def __init__(self):
        super().__init__(1)

    def generateData(self, a, b, data_points):
        self.dataByVariable.clear()
        for i in range(data_points):
            self.dataByVariable.append(np.random.beta(a, b))

    def plotData(self, plotType):
        if plotType == "raw_data":
            plt.plot(self.dataByVariable, "go")
            plt.show()
        elif plotType == "distribution":
            plt.hist(self.dataByVariable)
            plt.show()


