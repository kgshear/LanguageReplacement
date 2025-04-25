import math
import random

import pandas as pd
import config
from googletrans import Translator
translator = Translator()

class SentenceGoals():
    # beginner, intermediate, advanced
    percentage_bins = {0:[0,33], 1:[34,66], 2:[67,100]}
    current_category = 1

    def __init__(self):
        pass

    def calculate_num_to_replace(self, sentence):
        # calculates the num of words in a sentence u need to replace for a given category

        num_words = len(sentence.split())

        # [(number of spanish words, percentage spanish)]
        percentages_pairs = [(x, x/num_words) for x in range(num_words)]
        percentages_pairs.append((num_words,num_words))

        # if it's in the category we want, keep it in the list
        percentage_range = self.percentage_bins[self.current_category]
        percent_goal_match = [x for x in percentages_pairs if x[1] in range(percentage_range[0], percentage_range[1]+1)]

        # randomly pick an amount to replace that is in the category we want
        chosen_percentage = random.choice(percent_goal_match)
        num_to_replace = chosen_percentage[0]

        # next time this method is called, it will be another category
        if self.current_category == 2:
            self.current_category = 0
        else:
            self.current_category += 1
        return num_to_replace

    def get_category(self):
        return self.current_category


def get_dataframe():
    df = pd.read_csv(config.RAW_DATA_PATH, delimiter="\t", index_col=0, names=["language", "sentence"])
    df.drop(columns=["language"], inplace=True)
    return df

def calculate_percent_replaced(original_words, spanish_words):
    percent_replaced = (len(spanish_words) / len(original_words)) * 100
    return percent_replaced

def translate_snippet(sentence, goals: SentenceGoals):
    print("old sentence", sentence)
    words = sentence.split()
    category = goals.current_category
    num_to_replace = goals.calculate_num_to_replace(sentence)
    indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
    spanish_vocab = []
    for i in indices_to_replace:
        try:
            translation = translator.translate(words[i], src='en', dest='es')
            spanish_vocab.append(translation.text)
            words[i] = translation.text
        except Exception as e:
            print(f"Translation error at word '{words[i]} in sentence {words}': {e}")
            continue
    percent_replaced = calculate_percent_replaced(words, spanish_vocab)
    print("new sentence", ' '.join(words))
    return ' '.join(words), spanish_vocab, percent_replaced, category

