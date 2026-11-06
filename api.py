import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model import SmallAI

app = FastAPI(title="My Trained AI")

# 載入文字轉數字表
with open("word_to_id.json", encoding="utf-8") as f:
    word_to_id = json.load(f)

# 載入答案對照表
with open("id_to_answer.json", encoding="utf-8") as f:
    id_to_answer = json.load(f)
id_to_answer = {int(key): value for key, value in id_to_answer.items()}

# 載入你訓練好的模型
model = SmallAI(len(word_to_id))
model.load_state_dict(torch.load("my_ai_model.pt", map_location="cpu"))
model.eval()


def encode(question, max_len=10):
    ids = [
        word_to_id.get(word, 1)
        for word in question.lower().replace("?", "").split()
    ]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(item: Question):
    with torch.no_grad():
        x = torch.tensor([encode(item.question)], dtype=torch.long)
        prediction = model(x)
        answer_id = int(prediction[:, -1, :].argmax(dim=-1))
        answer = id_to_answer[answer_id]

    return {"answer": answer}
