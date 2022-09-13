from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime


def valid_and_save(valid_dataloader: DataLoader, device: torch.device, model, loss_fn, epoch: int, path: str) -> Tuple[
    float, float]:
    """"""
    total_avg_loss = 0
    total_correct = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_dataloader)):
            sents, labels, len_per_sent = batch
            sents = sents.to(device)
            labels = labels.squeeze(-1).to(device)  # B
            # print("labels:")
            # print(labels)
            logits = model(sents, len_per_sent.squeeze(-1).to(device))
            # print("logits:")
            # print(logits)
            loss = loss_fn(logits, labels)
            # print("loss")
            # print(loss)
            # calculate the accuracy
            pred_labels = torch.max(logits, dim=1)[1]
            # print(pred_labels)
            correct = torch.eq(pred_labels, labels).sum().item()
            total_avg_loss += loss.item()
            total_correct += correct

    # save to path specified
    torch.save(model, f"{path}/epoch{epoch}-{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}-ckpt.pth")

    return total_avg_loss, total_correct


if __name__ == '__main__':
    with open("./ckpt/epoch0-09-12-2022-20-49-19-ckpt.pth", "w") as f:
        pass
