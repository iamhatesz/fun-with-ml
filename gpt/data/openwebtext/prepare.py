from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    raw_dataset = load_dataset("openwebtext")
    dataset = raw_dataset["train"].train_test_split(
        test_size=0.0005, seed=2137, shuffle=True
    )
    dataset["val"] = dataset.pop("test")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def process(example):
        ids = tokenizer(example["text"])["input_ids"]
        ids.append(tokenizer.eos_token_id)
        out = {"ids": ids, "len": len(ids)}
        return out

    tokenized = dataset.map(
        process, remove_columns=["text"], desc="tokenizing", num_proc=12
    )

    for split, data in tokenized.items():
        total_len = np.sum(data["len"])
        filename = Path(__file__).parent / f"{split}.bin"
        arr = np.memmap(
            str(filename.absolute()), dtype=np.uint16, mode="w+", shape=(total_len,)
        )

        print("Writing:", filename)
        i = 0
        for example in tqdm(data):
            arr[i : i + example["len"]] = example["ids"]
            i += example["len"]
            if i % 100_000 == 0:
                arr.flush()
        arr.flush()
