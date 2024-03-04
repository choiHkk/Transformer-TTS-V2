# https://huggingface.co/learn/nlp-course/chapter6/5

import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Union

import torch
from tqdm.auto import tqdm

from configs.cross_entropy_hf import TransformerTTSv3Config
from encodec_wrapper import EncodecWrapper

seed = 1234
torch.manual_seed(seed)
random.seed(seed)

"""
tokenization strategy: sliding window
1) token-level length = sample-level length // hop length(320)

2) bpe-level length = token-level length // bpe hop length(hyper parameter)
* 16kHz, 16 bit depth, 1 sec
---> audio bits/s: 16000 * 16 * 1 = 256000bits/s
---> encodec bits/s: 400bits/s (basetts)
---> 256000 / 400 = 640
---> 320(hop length) * bit hop length = 640
---> bit hop length = 2
"""


def sliding_window(sequence: List[int], win_length: int = 2, hop_length: int = 2):
    windows = []
    for i in range(0, len(sequence) - win_length + 1, hop_length):
        window = sequence[i : i + win_length]
        windows.append(list(map(str, window)))
    return windows


def compute_pair_freqs(splits: Dict[str, List[int]], code_freqs: Dict[str, int]):
    pair_freqs = defaultdict(int)
    for code, freq in code_freqs.items():
        split = splits[code]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(
    a: str, b: str, splits: Dict[str, List[int]], code_freqs: Dict[str, int]
):
    for code in code_freqs:
        split = splits[code]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # split = split[:i] + [a + b] + split[i + 2 :]
                split = split[:i] + [" ".join([a, b])] + split[i + 2 :]
            else:
                i += 1
        splits[code] = split
    return splits


def generate_codes_vocab(
    codes_list: List[List[int]],
    win_length: int = 2,
    hop_length: int = 2,
    speech_code_vocab: List[Union[str]] = [],
    max_speech_code_vocab_size: int = 32768,
):
    merges = {}
    if len(speech_code_vocab) > 0:
        speech_code_vocab = list(map(str, speech_code_vocab))

    for codes in tqdm(codes_list, total=len(codes_list)):
        code_pairs = sliding_window(codes, win_length=win_length, hop_length=hop_length)
        code_freqs = Counter(tuple(window) for window in code_pairs)

        splits = {codes: list(codes) for codes in code_freqs.keys()}

        boolean = len(speech_code_vocab) < max_speech_code_vocab_size
        while boolean:
            pair_freqs = compute_pair_freqs(splits, code_freqs)
            if len(pair_freqs) == 0:
                break

            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            splits = merge_pair(*best_pair, splits, code_freqs)

            merged_pair = " ".join(best_pair)
            merges[best_pair] = merged_pair
            speech_code_vocab.append(merged_pair)

    # speech_code_vocab = sorted(set(speech_code_vocab))
    return speech_code_vocab, merges


def tokenize(
    codes: List[int], merges: Dict[str, str], win_length: int = 2, hop_length: int = 2
):
    tokens = sliding_window(codes, win_length=win_length, hop_length=hop_length)
    for pair, merge in merges.items():
        for idx, token in enumerate(tokens):
            i = 0
            while i < len(token) - 1:
                if token[i] == pair[0] and token[i + 1] == pair[1]:
                    token = token[:i] + [merge] + token[i + 2 :]
                else:
                    i += 1
            tokens[idx] = token
    tokens = [" ".join(x) for x in tokens]
    return tokens


def tokens_to_ids(tokens: List[str], speech_code_vocab: List[str]):
    token_id_table = {token: i for i, token in enumerate(speech_code_vocab)}
    # ids = [token_id_table[str(token)] for token in tokens]
    ids = []
    for token in tokens:
        if token_id_table.get(token) is not None:
            ids += [token_id_table[token]]
        else:
            ids += [token_id_table[t] for t in token.split(" ")]
    return ids


if __name__ == "__main__":
    model_config = TransformerTTSv3Config()
    encodec = EncodecWrapper()

    n_samples = 1000
    win_length = 2
    hop_length = 2

    code_freqs = defaultdict(int)
    n_codes = encodec.model.quantizer.bins

    # basetts: wavlm codebook size 256, bpe vocab size 8192 -> 8192 / 256 = 32
    max_speech_code_vocab_size = n_codes * model_config.c_vocab  # 1024 * 32

    speech_code_vocab = [i for i in range(encodec.model.quantizer.bins)]  # 0 ~ 1023

    # generate raw waveform samples
    T_dec = model_config.sampling_rate
    y_list = []
    for i in range(n_samples):
        y = 2 * torch.randn(1, model_config.sampling_rate * random.randint(1, 10)) - 1
        y_list.append(y)
    # y_list = y_list + y_list
    random.shuffle(y_list)

    # generate codes from raw waveform samples
    codes_list = []
    for y in tqdm(y_list, total=len(y_list)):
        with torch.no_grad():
            # codes = encodec.encode(y, return_emb=False)
            # codes = codes[:, : 1, :]
            codes = torch.randint(0, n_codes, (1, 1, y.size(-1) // 320))
        codes_list.append(codes.view(-1).tolist())

    vocab_path = "./vocab.json"
    if os.path.isfile(vocab_path):
        with open(vocab_path, "r", encoding="utf8") as j:
            data = json.loads(j.read())
            speech_code_vocab, merges = data["speech_code_vocab"], eval(data["merges"])
            win_length = data["win_length"]
            hop_length = data["hop_length"]
    else:
        print("generating vocabs")
        speech_code_vocab, merges = generate_codes_vocab(
            codes_list=codes_list,
            speech_code_vocab=speech_code_vocab,
            max_speech_code_vocab_size=max_speech_code_vocab_size,
            win_length=win_length,
            hop_length=hop_length,
        )
        data = {
            "win_length": win_length,
            "hop_length": hop_length,
            "speech_code_vocab": speech_code_vocab,
            "merges": str(merges),
        }
        with open(vocab_path, "w", encoding="utf8") as j:
            json.dump(data, j, ensure_ascii=False, indent=4, sort_keys=False)

    print(len(speech_code_vocab))

    for codes in tqdm(codes_list, total=len(codes_list)):
        tokens = tokenize(codes, merges, win_length=win_length, hop_length=hop_length)
        ids = tokens_to_ids(tokens, speech_code_vocab)
        print(len(codes), len(tokens), len(ids))
