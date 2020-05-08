import matplotlib.pyplot as plt


def plot_depth():
    depths = [1, 5, 15, 30]
    fold1_accuracies = [0.805800, 0.805800, 0.755400, 0.805800]
    fold2_accuracies = [0.841700, 0.791400, 0.784200, 0.812900]
    fold4_accuracies = [0.769800, 0.812900, 0.805800, 0.791400]
    accuracies = fold1_accuracies + fold2_accuracies + fold4_accuracies
    fold1_f1s = [0.810300, 0.808700, 0.754770, 0.810300]
    fold2_f1s = [0.839750, 0.791230, 0.784870, 0.819550]
    fold4_f1s = [0.766950, 0.816250, 0.810100, 0.798800]
    f1s = fold1_f1s + fold2_f1s + fold4_f1s

    fig, axes = plt.subplots()
    plt.title("Accuracy | Prune Tree By Depth")
    axes.plot(depths, fold1_accuracies, label="fold1", marker='o', markersize=2)
    axes.plot(depths, fold2_accuracies, label="fold2", marker='o', markersize=2)
    axes.plot(depths, fold4_accuracies, label="fold4", marker='o', markersize=2)
    axes.set(xlabel='Max Tree Depth', ylabel='Accuracy')
    axes.set_xlim(0, max(depths) * 1.1)
    axes.set_ylim(min(accuracies) * 0.9, max(accuracies) * 1.1)
    axes.grid()
    axes.legend()
    fig.savefig("../output/figures/depth_accuracy.png")

    fig, axes = plt.subplots()
    plt.title("Micro-F1 | Prune Tree By Depth")
    axes.plot(depths, fold1_f1s, label="fold1", marker='o', markersize=2)
    axes.plot(depths, fold2_f1s, label="fold2", marker='o', markersize=2)
    axes.plot(depths, fold4_f1s, label="fold4", marker='o', markersize=2)
    axes.set(xlabel='Max Tree Depth', ylabel='Mirco-F1')
    axes.set_xlim(0, max(depths) * 1.1)
    axes.set_ylim(min(f1s) * 0.9, max(f1s) * 1.1)
    axes.grid()
    axes.legend()
    fig.savefig("../output/figures/depth_f1.png")


def plot_width():
    widths = [5, 10, 20, 50, 100, 150]
    fold1_accuracies = [0.791400, 0.697800, 0.546800, 0.503600, 0.503600, 0.237400]
    fold2_accuracies = [0.719400, 0.733800, 0.554000, 0.489200, 0.366900, 0.323700]
    fold4_accuracies = [0.733800, 0.669100, 0.568300, 0.438800, 0.395700, 0.352500]
    accuracies = fold1_accuracies + fold2_accuracies + fold4_accuracies
    fold1_f1s = [0.785730, 0.687600, 0.536100, 0.494900, 0.474900, 0.194100]
    fold2_f1s = [ 0.716150, 0.739000, 0.557880, 0.470680, 0.270650, 0.300330]
    fold4_f1s = [0.736670, 0.672970, 0.573400, 0.426400, 0.360730, 0.293500]
    f1s = fold1_f1s + fold2_f1s + fold4_f1s

    fig, axes = plt.subplots()
    plt.title("Accuracy | Prune Tree By Width")
    axes.plot(widths, fold1_accuracies, label="fold1", marker='o', markersize=2)
    axes.plot(widths, fold2_accuracies, label="fold2", marker='o', markersize=2)
    axes.plot(widths, fold4_accuracies, label="fold4", marker='o', markersize=2)
    axes.set(xlabel='Min Children Number', ylabel='Accuracy')
    axes.set_xlim(0, max(widths) * 1.1)
    axes.set_ylim(min(accuracies) * 0.9, max(accuracies) * 1.1)
    axes.grid()
    axes.legend()
    fig.savefig("../output/figures/width_accuracy.png")

    fig, axes = plt.subplots()
    plt.title("Micro-F1 | Prune Tree By Width")
    axes.plot(widths, fold1_f1s, label="fold1", marker='o', markersize=2)
    axes.plot(widths, fold2_f1s, label="fold2", marker='o', markersize=2)
    axes.plot(widths, fold4_f1s, label="fold4", marker='o', markersize=2)
    axes.set(xlabel='Min Children Number', ylabel='Micro-F1')
    axes.set_xlim(0, max(widths) * 1.1)
    axes.set_ylim(min(f1s) * 0.9, max(f1s) * 1.1)
    axes.grid()
    axes.legend()
    fig.savefig("../output/figures/width_f1.png")


if __name__ == "__main__":
    plot_depth()
    plot_width()
