import math
import random

import pandas as pd
import config
from googletrans import Translator
translator = Translator()

class SentenceGoals():
    # beginner, intermediate, advanced
    percentage_bins = {1:[0,33], 2:[34,66], 3:[67,100]}
    current_percentage_goal = 1

    def __init__(self):
        pass

    def calculate_num_to_replace(self, sentence):
        # calculates the num of words in a sentence u need to replace for a given category

        num_words = len(sentence.split())
        percentage_range = self.percentage_bins[self.current_percentage_goal]
        percentage = random.randint(percentage_range[0], percentage_range[1])
        num_to_replace = min(math.ceil(num_words * (percentage / 100)), num_words)

        if self.current_percentage_goal == 10:
            self.current_percentage_goal = 1
        else:
            self.current_percentage_goal += 1
        return num_to_replace


def get_dataframe():
    df = pd.read_csv(config.RAW_DATA_PATH, delimiter="\t", index_col=0, names=["language", "sentence"])
    df.drop(columns=["language"], inplace=True)
    return df

def calculate_percent_replaced(original_words, spanish_words):
    percent_replaced = (len(spanish_words) / len(original_words)) * 100
    return percent_replaced

def translate_snippet(sentence, goals: SentenceGoals):
    words = sentence.split()
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
    return ' '.join(words), spanish_vocab, percent_replaced

