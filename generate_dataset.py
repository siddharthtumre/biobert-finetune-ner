from datasets import load_dataset, DatasetDict


def main():
    dataset = load_dataset("jnlpba")

    test_valid = dataset['validation'].train_test_split(test_size=0.5)
    final_dataset = DatasetDict(
        {'train': dataset['train'], 'test': test_valid['test'], 'valid': test_valid['train']})

    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    id2label = {i: label for i, label in enumerate(label_names)}

    for split, data in final_dataset.items():
        with open(f"./datasets/NER/JNLPBA/{split}.txt.tmp", "w+") as fp:
            for row in data:

                for i in range(len(row["tokens"])):
                    fp.write(row["tokens"][i] + " " +
                             id2label[row["ner_tags"][i]])
                    fp.write("\n")

                fp.write("\n")


if __name__ == "__main__":
    main()
