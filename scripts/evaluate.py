"""Score the model on the toy world: % of answers that are correct.

For each val story, split at the last space:
  prompt = "tom has a red ball. ana has a blue ball. tom's ball is "
  answer = "red."
Let the model write letters until "." (always the most likely letter), compare with the answer.
Then do it again with the memory off (Gs = 0): the score should fall.
Usage: python -m scripts.evaluate checkpoints/tiny.pt
"""


def main():
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
