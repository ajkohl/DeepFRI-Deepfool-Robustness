import matplotlib.pyplot as plt
import json

# Load mutation thresholds from the provided JSON file
with open('output/mutation_thresholds.json', 'r') as file:
    mutation_thresholds = json.load(file)

# # Plotting function as described
# def plot_mutation_thresholds(mutation_thresholds):
#     plt.figure(figsize=(10, 6))
#     plt.hist(mutation_thresholds, bins=10, edgecolor='black')
#     plt.title('Distribution of Mutation Thresholds for Misclassification')
#     plt.xlabel('Number of Mutations')
#     plt.ylabel('Frequency')
#     plt.savefig('output/plot.png')
#     plt.show()

# # Call the plotting function
# plot_mutation_thresholds(mutation_thresholds)

def plot_mutation_thresholds_with_edges(mutation_thresholds, bin_edges):
    plt.figure(figsize=(10, 6))
    plt.hist(mutation_thresholds, bins=bin_edges, edgecolor='black')  # Specify bin edges directly
    plt.title('Distribution of Mutation Thresholds for Misclassification')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Frequency')
    plt.savefig('output/plot.png')
    plt.show()

# Define your custom bin edges
bin_edges = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]

# Call the plotting function with custom bin edges
plot_mutation_thresholds_with_edges(mutation_thresholds, bin_edges)



# import matplotlib.pyplot as plt
# import json
# import numpy as np

# # Load mutation thresholds from the provided JSON file
# with open('output/mutation_thresholds.json', 'r') as file:
#     mutation_thresholds = json.load(file)

# # Plotting function with dynamic bin sizing using the Freedman-Diaconis rule
# def plot_mutation_thresholds(mutation_thresholds):
#     plt.figure(figsize=(10, 6))
    
#     # Calculate bin width using the Freedman-Diaconis rule
#     IQR = np.percentile(mutation_thresholds, 75) - np.percentile(mutation_thresholds, 25)
#     bin_width = 2 * IQR / len(mutation_thresholds) ** (1/3)
#     bins = np.arange(min(mutation_thresholds), max(mutation_thresholds) + bin_width, bin_width)

#     plt.hist(mutation_thresholds, bins=bins, edgecolor='black')
#     plt.title('Distribution of Mutation Thresholds for Misclassification')
#     plt.xlabel('Number of Mutations')
#     plt.ylabel('Frequency')
#     plt.savefig('output/plot.png')
#     plt.show()

# # Call the plotting function with dynamic bins
# plot_mutation_thresholds(mutation_thresholds)
