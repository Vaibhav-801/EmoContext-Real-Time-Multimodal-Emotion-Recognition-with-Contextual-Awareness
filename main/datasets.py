import pandas as pd
from fsspec.utils import tokenize
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import torch
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

dev_df=pd.read_csv("/mnt/d/EmoBase/Data/dev.txt",sep="\t")
test_df=pd.read_csv("/mnt/d/EmoBase/Data/test.txt",sep="\t")
train_df=pd.read_csv("/mnt/d/EmoBase/Data/train.txt",sep="\t")

train_df.isnull().sum()
dev_df.isnull().sum()
test_df.isnull().sum()
test_df.dropna()
test_df.reset_index(drop=True)
train_df.reset_index(drop=True)
dev_df.reset_index(drop=True)

train_df = train_df.dropna()
dev_df = dev_df.dropna()
test_df = test_df.dropna()

train_df["text"]=train_df["turn1"]+" [SEP] "+train_df["turn2"]+" [SEP] "+train_df["turn3"]
test_df["text"]=test_df["turn1"]+" [SEP] "+test_df["turn2"]+" [SEP] "+test_df["turn3"]
dev_df["text"]=dev_df["turn1"]+" [SEP] "+dev_df["turn2"]+" [SEP] "+dev_df["turn3"]

emotion2label = {
    "others": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3
}

train_df["label_id"]=train_df["label"].map(emotion2label)

dev_df["label_id"]=dev_df["label"].map(emotion2label)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# device = torch.device('cuda')
# model = model.to(device)

encoded_train = tokenizer(
    train_df["text"].tolist(),
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
encoded_train["input_ids"].shape
encoded_dev=tokenizer(dev_df["text"].tolist(),max_length=128,return_tensors="pt",padding="max_length",truncation=True)
encoded_test=tokenizer(test_df["text"].tolist(),max_length=128,return_tensors="pt",padding="max_length",truncation=True)

import torch
train_labels=torch.tensor(train_df["label_id"].tolist())
dev_labels=torch.tensor(dev_df["label_id"].tolist())
print(train_labels[:5])

test_df["label_id"] = test_df["label"].map(emotion2label)
test_labels = torch.tensor(test_df["label_id"].tolist())

from torch.utils.data import Dataset
class EmoDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }
        return item

test_dataset=EmoDataset(encoded_test,test_labels)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
train_dataset=EmoDataset(encoded_train,train_labels)
dev_dataset=EmoDataset(encoded_dev,dev_labels)
test_dataset = EmoDataset(encoded_test, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=32,shuffle=False)

y=train_labels.numpy()
class_weight=compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1,2,3,]),
    y=y
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

#this is the code ill use for train my data


# from transformers import get_linear_schedule_with_warmup
# total_steps = len(train_loader) * 5
# warmup_steps = total_steps // 10
# optimizer = AdamW(model.parameters(), lr=2e-5)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=total_steps
# )
# class_weight_tensor = torch.tensor(class_weight, dtype=torch.float).to(device)
# loss_fn = CrossEntropyLoss(weight=class_weight_tensor)
# for epoch in range(7):
#     print(f"Epoch {epoch+1}/7")
#     model.train()
#     for batch in train_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#         optimizer.zero_grad()
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         logits = outputs.logits
#         loss = loss_fn(logits, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#this the main model train part

model.eval()
all_pred=[]
all_labels=[]

with torch.no_grad():
    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs=model(input_ids=input_ids,attention_mask=attention_mask)
        logits=outputs.logits

        pred=torch.argmax(logits,dim=-1)

        all_pred.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels,all_pred))

model.eval()
all_pred=[]
all_labels=[]
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        all_pred.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_pred))