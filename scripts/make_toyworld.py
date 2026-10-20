"""Generate the toy world: short stories whose answer is only in the story.

Usage: python -m scripts.make_toyworld --level 1 --stories 10000
Writes data/toyworld_l1_train.txt and data/toyworld_l1_val.txt (one story per line).
"""

import argparse
import os
import random

NAMES = ["tom", "ana", "leo", "mia", "sam", "eva", "max", "zoe", "ben", "ida",
         "dan", "amy", "joe", "liz", "ian", "kim", "bob", "sue", "ned", "ava"]
COLORS = ["red", "blue", "green", "pink", "gray", "black", "white", "brown", "gold", "teal"]
OBJECTS = ["ball", "cup", "hat", "box", "car", "pen", "bag", "key"]
FRUITS = ["banana", "pear", "plum", "kiwi", "lime", "fig"]   # FRUITS[0] makes you happy


def level1():
    # two people, one ball each, random colors: the answer is only in the story
    a, b = random.sample(NAMES, 2)
    ca = random.choice(COLORS)
    cb = random.choice(COLORS)
    who, color = random.choice([(a, ca), (b, cb)])
    return f"{a} has a {ca} ball. {b} has a {cb} ball. {who}'s ball is {color}."


def level2():
    # three people, three different objects: more distraction
    people = random.sample(NAMES, 3)
    objects = random.sample(OBJECTS, 3)
    facts = [(p, o, random.choice(COLORS)) for p, o in zip(people, objects)]
    text = " ".join(f"{p} has a {c} {o}." for p, o, c in facts)
    p, o, c = random.choice(facts)
    return f"{text} {p}'s {o} is {c}."


def level3():
    # rule (same in every story): banana makes you happy  -> learned in the G's
    # fact (only this story): who ate the banana          -> stored in sigma
    a, b = random.sample(NAMES, 2)          # a eats the banana
    other = random.choice(FRUITS[1:])
    lines = [f"{a} eats a {FRUITS[0]}.", f"{b} eats a {other}."]
    random.shuffle(lines)
    return f"{lines[0]} {lines[1]} who is happy? {a}."


GENERATORS = {1: level1, 2: level2, 3: level3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--stories", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    make = GENERATORS[args.level]
    # dict.fromkeys removes duplicates, so no story is in both train and val
    stories = list(dict.fromkeys(make() for _ in range(args.stories)))

    n_val = len(stories) // 10
    os.makedirs("data", exist_ok=True)
    for split, part in [("val", stories[:n_val]), ("train", stories[n_val:])]:
        path = f"data/toyworld_l{args.level}_{split}.txt"
        with open(path, "w") as f:
            f.write("\n".join(part) + "\n")
        print(f"wrote {len(part)} stories to {path}")

    print("example:", stories[0])


if __name__ == "__main__":
    main()
