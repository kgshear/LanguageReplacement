import utils
import config

def parse_sentences(df):
    goals = utils.SentenceGoals()
    with open(config.OUTPUT_DIR, "w", encoding="utf-8") as f:
        f.write("")
        for idx, row in df.iterrows():
            original = row["sentence"]
            f.write(f"Original: {original}\n")
            translated, spanish_vocab, percent_replaced = utils.translate_snippet(original, goals)

            f.write(f"Original: {original}\n")
            f.write(f"Translated: {translated}\n")
            f.write(f"Percent Replaced: {percent_replaced:.2f}%\n")
            f.write(f"Spanish Vocab: {', '.join(spanish_vocab)}\n")
            f.write("=" * 50 + "\n")


if __name__ == "__main__":
    df = utils.get_dataframe()
    parse_sentences(df)