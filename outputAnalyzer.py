import matplotlib.pyplot as plt
import numpy as np

fileToAnalyze = "NBfaces.txt"

with open(fileToAnalyze, "r") as f:
    training_times = {}
    accuracies = {}
    current_training_set_size = None

    for line in f:
        if "training set size" in line:
            current_training_set_size = int(line.split(":")[1].strip())
            if current_training_set_size not in training_times.keys():
                training_times[current_training_set_size] = []
                accuracies[current_training_set_size] = []
        elif "training time" in line.lower():
            training_time = float(line.split()[0])
            training_times[current_training_set_size].append(training_time)
        elif "correct out of 100" in line:
            accuracy = float(line.split()[0])
            accuracies[current_training_set_size].append(accuracy)

# Calculate average training time and accuracy for each training set size
average_training_times = []
average_accuracies = []
std_training_times = []
std_accuracies = []
training_set_sizes = sorted(list(training_times.keys()))

# print(accuracies)

for training_set_size in training_set_sizes:
    average_training_times.append(np.mean(training_times[training_set_size]))
    average_accuracies.append(np.mean(accuracies[training_set_size]))
    std_training_times.append(np.std(training_times[training_set_size]))
    std_accuracies.append(np.std(accuracies[training_set_size]))


# Create four subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
(ax1, ax2), (ax3, ax4) = axs

# Plot average training time on the first subplot
ax1.plot(training_set_sizes, average_training_times, 'o-')
ax1.set_ylabel("Average Training Time (s)")
ax1.set_xlabel("Training Set Size")
ax1.set_title("Average Training Time")

# Plot average accuracy on the second subplot
ax2.plot(training_set_sizes, average_accuracies, 'o-')
ax2.set_ylabel("Average Accuracy (%)")
ax2.set_xlabel("Training Set Size")
ax2.set_title("Average Accuracy")

# Plot standard deviation of training time on the third subplot
ax3.plot(training_set_sizes, std_training_times, 'o-')
ax3.set_ylabel("Standard Deviation of Training Time (s)")
ax3.set_xlabel("Training Set Size")
ax3.set_title("Standard Deviation of Training Time")

# Plot standard deviation of accuracy on the fourth subplot
ax4.plot(training_set_sizes, std_accuracies, 'o-')
ax4.set_ylabel("Standard Deviation of Accuracy (%)")
ax4.set_xlabel("Training Set Size")
ax4.set_title("Standard Deviation of Accuracy")

for ax in axs.flatten():
    ax.grid(True)
    ax.set_xticks(training_set_sizes)
    ax.set_xticklabels(training_set_sizes)

plt.show()