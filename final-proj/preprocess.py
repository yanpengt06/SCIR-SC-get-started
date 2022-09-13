import json
import os
from collections import Counter
from typing import Tuple, Dict

import jieba
import pandas as pd
from tqdm import tqdm


def clean_sent(sent: str) -> str:
    return sent.replace("\n", "")


def encode_and_pad(
        input_sents: list, word_map: Dict[str, int], word_limit: int
) -> Tuple[list, list]:
    """
    Encode sentences, and pad them to fit word_limit.

    Parameters
    ----------
    input_sents : list
        Sentences ([ word1, ..., wordn ])

    word_map : Dict[str, int]
        Word2ix map

    word_limit : int
        Max number of words in a sentence

    Returns
    -------
    encoded_sents : list
        Encoded and padded sentences

    words_per_sentence : list
        Number of words per sentence
    """
    encoded_sents = list(
        map(lambda s: list(
            map(lambda w: word_map.get(w, word_map['<unk>']), s)
        ) + [0] * (word_limit - len(s)), input_sents)
    )
    words_per_sentence = list(map(lambda s: len(s), input_sents))
    return encoded_sents, words_per_sentence


def read_from_csv(csv_folder: str, split: str, word_limit: int) -> Tuple[list, list, Counter]:
    """

    Parameters
    ----------
    csv_folder      /data/restaurant
    split           train/valid/test
    word_limit      int: minimum word freq

    Returns
    -------
    sents: [[’今天‘，’是’，‘中秋节’],[‘嗯’，‘是的’],[],[]]
    labels: [1,2,0,3]
    word_counter
    """
    sents = []
    labels = []
    word_counter = Counter()
    df = pd.read_csv(f"{csv_folder}/{split}.csv")
    for i in tqdm(range(df.shape[0])):
        sentence = df.loc[i, "content"]
        label = df.loc[i, "service_waiters_attitude"]  # waiter_attitude
        # clean sentence
        sentence = clean_sent(sentence)[1:-1]
        sentence = list(jieba.cut(sentence, cut_all=False))

        # limit tokens
        end = -1 if len(sentence) <= word_limit else word_limit
        sentence = sentence[:end]
        word_counter.update(sentence)
        sents.append(sentence)
        labels.append(label)

    return sents, labels, word_counter


def run_prepro(
        csv_folder: str, output_folder: str, word_limit: int, min_word_count: int = 10
) -> Tuple[list, list, list, list, list, list, dict]:
    """
    1. tokenize sentence and construct the word_map.
    2. read train, valid, test corpus.
    Parameters
    ----------
    csv_folder : str
        Folder where the CSVs with the raw data are located

    output_folder : str
        Folder where files must be created

    word_limit : int
        Truncate long sentences to these many words

    min_word_count : int
        Discard rare words which occur fewer times than this number

    Returns
    -------
    train_sents, train_labels, valid_sents, valid_labels, test_sents, test_labels, word_map
    """
    # --------------------- training data ---------------------
    print('\nTraining data: reading and preprocessing...\n')
    train_sents, train_labels, word_counter = read_from_csv(csv_folder, 'train', word_limit)

    # create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print('\nTraining data: discarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
        min_word_count, len(word_map)))
    # save word map
    with open(f"{output_folder}/word_map.json", 'w', encoding="utf-8") as j:
        json.dump(word_map, j, ensure_ascii=False)
    print('Training data: word map saved to %s.\n' % os.path.abspath(output_folder))

    # # encode and pad
    # print('Training data: encoding and padding...\n')
    # encoded_train_sents, words_per_train_sent = encode_and_pad(train_sents, word_map, word_limit)

    # save
    # print('Training data: saving...\n')
    assert len(train_sents) == len(train_labels)
    # # because of the large data, saving as a JSON can be very slow
    # torch.save({
    #     'sents': encoded_train_sents,
    #     'labels': train_labels,
    #     'words_per_sentence': words_per_train_sent
    # }, f"{output_folder}/TRAIN_data.pth.tar")
    # print('Training data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    # free some memory
    # del train_sents, encoded_train_sents, train_labels, words_per_train_sent

    # --------------------- valid data ---------------------
    print('Valid data: reading and preprocessing...\n')
    valid_sents, valid_labels, _ = read_from_csv(csv_folder, 'valid', word_limit)

    # --------------------- test data ---------------------
    print('Test data: reading and preprocessing...\n')
    test_sents, test_labels, _ = read_from_csv(csv_folder, 'test', word_limit)

    # # encode and pad
    # print('\nTest data: encoding and padding...\n')
    # encoded_test_sents, words_per_test_sent = encode_and_pad(test_sents, word_map, word_limit)

    # save
    # print('Test data: saving...\n')
    # assert len(encoded_test_sents) == len(test_labels) == len(words_per_test_sent)
    # torch.save({
    #     'sents': encoded_test_sents,
    #     'labels': test_labels,
    #     'words_per_sentence': words_per_test_sent
    # }, os.path.join(output_folder, 'TEST_data.pth.tar'))
    # print('Test data: encoded, padded data saved to %s.\n' % os.path.abspath(output_folder))

    print('All done!\n')
    return train_sents, train_labels, valid_sents, valid_labels, test_sents, test_labels, word_map


