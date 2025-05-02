import utils
import config
import pandas as pd
import asyncio



async def parse_sentences(df):
    goals = utils.SentenceGoals()
    rows = []

    for idx, row in df.iterrows():
        original = row["sentence"]
        translation, spanish_vocab, percent_replaced, category = await utils.translate_snippet(original, goals)
        rows.append([original, translation, spanish_vocab, percent_replaced, category])

    trans_df = pd.DataFrame(rows, columns=["original", "translation", "vocab", "percentage_replaced", "category"])
    json_string = trans_df.to_json(orient="records")
    with open(config.REPLACEMENTS_PATH_JSON, "w", encoding="utf-8") as f:
        f.write(json_string)

    csv_string = trans_df.to_csv()
    with open(config.REPLACEMENTS_PATH_CSV, "w", encoding="utf-8") as f:
        f.write(csv_string)

async def parse_qa_pairs(df):
    goals = utils.SentenceGoals()
    rows = []

    for idx, row in df.iterrows():
        question, answer, spanish_vocab, percent_replaced, spanish_vocab, category = \
            await utils.translate_qa_snippet(row["question"], row["answer"], goals)
        rows.append([question, answer, spanish_vocab, percent_replaced, spanish_vocab, category])

    trans_df = pd.DataFrame(rows, columns=["question", "answer", "spanish_vocab", "percent_replaced",
                                           "spanish_vocab", "category"])
    json_string = trans_df.to_json(orient="records")
    with open(config.QA_OUTPUT_PATH_JSON, "w", encoding="utf-8") as f:
        f.write(json_string)

    csv_string = trans_df.to_csv()
    with open(config.QA_OUTPUT_PATH_CSV, "w", encoding="utf-8") as f:
        f.write(csv_string)

if __name__ == "__main__":
    # df = utils.get_dataframe()
    # df = df.dropna()
    # df = utils.save_qa_csv()
    df = utils.get_qa_dataframe()
    asyncio.run(parse_qa_pairs(df))
    # df = df.sample(3000)

    # asyncio.run(parse_sentences(df))
