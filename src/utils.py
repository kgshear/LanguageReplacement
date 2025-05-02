import json
import math
import random

import pandas as pd
import config
from googletrans import Translator
from nltk.stem import WordNetLemmatizer
import spacy

translator = Translator()
nlp = spacy.load("es_core_news_sm")


class SentenceGoals():
    # beginner, intermediate, advanced
    percentage_bins = {0: [0, .33], 1: [.34, .66], 2: [.67, 1]}
    current_category = 0

    def __init__(self):
        pass

    def calculate_num_to_replace(self, sentence):
        # calculates the num of words in a sentence u need to replace for a given category

        num_words = len(sentence.split())

        # [(number of spanish words, percentage spanish)]
        percentages_pairs = [(x, x / num_words) for x in range(num_words)]
        percentages_pairs.append((num_words, num_words / num_words))
        # print("percentage pairs", percentages_pairs)

        # if it's in the category we want, keep it in the list
        percentage_range = self.percentage_bins[self.current_category]
        percent_goal_match = [x for x in percentages_pairs if percentage_range[0] <= x[1] <= percentage_range[1] + .01]

        # randomly pick an amount to replace that is in the category we want
        chosen_percentage = random.choice(percent_goal_match)
        num_to_replace = chosen_percentage[0]
        # print("amount to replace", num_to_replace)

        # next time this method is called, it will be another category
        return num_to_replace

    def get_category(self):
        return self.current_category

    def update_category(self):
        if self.current_category == 2:
            self.current_category = 0
        else:
            self.current_category += 1


def get_dataframe():
    df = pd.read_csv(config.RAW_DATA_PATH, delimiter="\t", names=["num", "language", "sentence"])
    df.drop(columns=["num", "language"], inplace=True)
    return df


def save_qa_csv():
    qa_json = json.load(open(config.RAW_QA_JSON_PATH))
    records = []
    for entry in qa_json["data"]:
        q_map = {q["turn_id"]: q["input_text"] for q in entry["questions"]}
        for ans in entry['answers']:
            if ans.get('bad_turn') == "true":
                continue
            turn_id = ans['turn_id']
            question = q_map.get(turn_id)
            answer = ans["input_text"]
            if question is not None and answer is not None:
                records.append({'question': question, 'answer': answer})
    df = pd.DataFrame(records)
    df.to_csv(config.RAW_QA_CSV_PATH)
    return df


def get_qa_dataframe():
    df = pd.read_csv(config.RAW_QA_CSV_PATH, delimiter=",", names=["question", "answer"])
    df = df[df['answer'].str.split().str.len() >= 3]
    df = df[df['question'].str.split().str.len() >= 3]
    df = df.sample(3000)
    return df


def calculate_percent_replaced(original_words, spanish_words):
    percent_replaced = (len(spanish_words) / len(original_words)) * 100
    return percent_replaced


def clean_spanish_vocab(spanish_vocab_list):
    temp_text = " ".join(spanish_vocab_list)
    special_characters = "!¡@#$%^&*()_+=-`~[]\\{}|;':\",./<>?¿"
    for char in special_characters:
        temp_text = temp_text.replace(char, "")
    words = temp_text.split()
    spanish_vocab_list = [word.lower() for word in words]
    # lemmatized_vocab = [nlp(word)[0].lemma_ for word in spanish_vocab_list]

    # removes duplicates but keeps it as a list
    return list(set(spanish_vocab_list))


def get_all_index_combos(words, max_phrase_len):
    combos = []
    for start in range(len(words)):
        for length in range(1, max_phrase_len + 1):
            if start + length <= len(words):
                combos.append(list(range(start, start + length)))
    return combos


def select_non_overlapping(combos, num_to_select):
    random.shuffle(combos)
    selected_combos = []
    used_indices = set()
    total_words = 0

    for combo in combos:
        if any(i in used_indices for i in combo):
            continue
        if total_words + len(combo) > num_to_select:
            continue
        selected_combos.append(combo)
        used_indices.update(combo)
        total_words += len(combo)
        if total_words >= num_to_select:
            break

    return selected_combos


def get_replacement_indexes(words, num_to_replace):
    indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
    return indices_to_replace


async def translate_snippet(sentence, goals: SentenceGoals):
    # print("old sentence", sentence)
    words = sentence.split()
    category = goals.current_category
    num_to_replace = goals.calculate_num_to_replace(sentence)
    goals.update_category()
    indices_to_replace = get_replacement_indexes(words, num_to_replace)
    spanish_vocab = []
    for i in indices_to_replace:
        try:
            translation = await translator.translate(words[i], src='en', dest='es')
            spanish_vocab.append(translation.text)
            words[i] = translation.text
        except Exception as e:
            print(f"Translation error at word '{words[i]} in sentence {words}': {e}")
            continue
    percent_replaced = calculate_percent_replaced(words, spanish_vocab)
    # print("new sentence", ' '.join(words))
    # print("old spanish_vocab", spanish_vocab)
    spanish_vocab = clean_spanish_vocab(spanish_vocab)
    # print("new spanish_vocab", spanish_vocab)
    return ' '.join(words), spanish_vocab, percent_replaced, category


async def translate_qa_snippet(question, answer, goals: SentenceGoals):
    # print("old question", question)
    # print("old answer", answer)
    words_q = question.split()
    words_a = answer.split()
    category = goals.current_category
    num_to_replace_q = goals.calculate_num_to_replace(question)
    num_to_replace_a = goals.calculate_num_to_replace(answer)
    percent_replaced = round((num_to_replace_a / len(words_a)) * 100, 2)
    # print(num_to_replace_q, num_to_replace_a)
    goals.update_category()
    combos_q = get_all_index_combos(words_q, max_phrase_len=num_to_replace_q)
    selected_combos_q = select_non_overlapping(combos_q, num_to_select=num_to_replace_q)
    combos_a = get_all_index_combos(words_a, max_phrase_len=num_to_replace_a)
    selected_combos_a = select_non_overlapping(combos_a, num_to_select=num_to_replace_a)
    spanish_vocab = []

    for combo in selected_combos_q:
        if max(combo) >= len(words_q):
            print(f"Skipping combo {combo} — exceeds sentence length {len(words_q)}")
            continue
        phrase = ' '.join(words_q[i] for i in combo)
        try:
            # print(phrase)
            translation = await translator.translate(phrase, src='en', dest='es')
            translated = translation.text
            words_q[combo[0]] = translated
            for i in combo[1:]:
                words_q[i] = ''
        except Exception as e:
            print(f"Translation error for '{phrase}': {e}")
    words_q = [w for w in words_q if w]

    for combo in selected_combos_a:
        if max(combo) >= len(words_a):
            print(f"Skipping combo {combo} — exceeds sentence length {len(words_a)}")
            continue
        phrase = ' '.join(words_a[i] for i in combo)
        try:
            # print(phrase)
            translation = await translator.translate(phrase, src='en', dest='es')
            translated = translation.text
            spanish_vocab = spanish_vocab + translated.split()
            words_a[combo[0]] = translated
            for i in combo[1:]:
                words_a[i] = ''
        except Exception as e:
            print(f"Translation error for '{phrase}': {e}")
    words_a = [w for w in words_a if w]

    # print("new question", ' '.join(words_q))
    # print("new answer", ' '.join(words_a))
    # print("old spanish_vocab", spanish_vocab)
    spanish_vocab = clean_spanish_vocab(spanish_vocab)
    # print("new spanish_vocab", spanish_vocab)
    return ' '.join(words_q), ' '.join(words_a), question, answer, percent_replaced, spanish_vocab, category
