import json

from utils import system_prompt

MAX_HISTORY = 6
LANGS = ("zh", "en")
CATEGORIES = ("knowledge", "emotion", "persona", "casual")


class DialogueDataset:
    def __init__(self, path):
        self.path = path

        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        for i, row in enumerate(self.rows):
            check(row, i)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return prompt(row), answer(row)


def check(row, i=0):
    msgs = row["messages"]

    assert msgs[0]["role"] == "system", f"Riga {i}: la prima frase non è system"
    assert msgs[-1]["role"] == "assistant", f"Riga {i}: l'ultima frase non è assistant"
    assert len(msgs) % 2 == 1, f"Riga {i}: dopo system non ci sono coppie user/assistant"

    for j, m in enumerate(msgs[1:]):
        want = "user" if j % 2 == 0 else "assistant"
        assert m["role"] == want, f"Riga {i} frase {j + 1}: dovrebbe essere {want}"
        assert isinstance(m["content"], str), f"Riga {i} frase {j + 1}: content non è una stringa"

    assert history_turns(row) <= MAX_HISTORY, f"Riga {i}: storia supera {MAX_HISTORY} turni"

    meta = row.get("meta", {})
    assert meta.get("lang") in LANGS, f"Riga {i}: lang non valido: {meta.get('lang')}"
    assert meta.get("category") in CATEGORIES, f"Riga {i}: category non valida: {meta.get('category')}"

    want = system_prompt(meta["category"], meta["lang"])
    assert msgs[0]["content"] == want, f"Riga {i}: system non corrisponde a category/lang"


def history_turns(row):
    return (len(row["messages"]) - 3) // 2


def question(row):
    return row["messages"][-2]["content"]


def answer(row):
    return row["messages"][-1]["content"]


def prompt(row):
    return row["messages"][:-1]


def load(path):
    return DialogueDataset(path).rows


if __name__ == "__main__":
    dataset = DialogueDataset("data/sample.jsonl")

    print("数据集大小：", len(dataset))

    msgs, ref = dataset[0]

    print("对话轮数：", len(msgs))
    print("提问：", msgs[-1]["content"][:40], "...")
    print("参考回答：", ref)
