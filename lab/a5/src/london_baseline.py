# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

# modify -2
from tqdm import tqdm

from src import utils

with open("../birth_dev.tsv", "r") as f:
    dev_examples = f.read().rstrip().split("\n")
    predictions = ["London"] * len(dev_examples)
total, correct = utils.evaluate_places("../birth_dev.tsv", predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))