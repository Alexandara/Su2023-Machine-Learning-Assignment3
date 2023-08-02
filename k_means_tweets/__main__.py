"""
Module runs and documents neural_network code.

@author Alexis Tudor
"""
from k_means import KMeansTweets
import matplotlib.pyplot as plt

def run(file, k=2):
    model = KMeansTweets("https://personal.utdallas.edu/~art150530/wsjhealth.txt", k=k)
    model.train()
    sse = model.sse()
    results_file.write(str(k) + "," + str(sse) + "," "Cluster sizes for k=" + str(k) + "\n")
    clusters = model.cluster()
    for cluster in clusters:
        results_file.write(" , ," + str(k) + ": " + str(len(cluster)) + "tweets\n")
    return model.test()

if __name__ == '__main__':
    results_file = open("results.csv", "a")
    results_file.write("K Value,SSE,Cluster Sizes\n")
    results = []
    for i in range(10):
        if i != 0:
            results.append(run(results_file, k=i))
    x_axis = []
    for i in range(len(results)):
        x_axis.append(i)
    plt.plot(x_axis, results)
    plt.ylabel('Average Distance to Center')
    plt.xlabel('K Value')
    plt.title("Distance to Center Per K Value")
    plt.savefig("k_values.png")
    results_file.close()