from typing import Tuple

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn


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


def valid_and_save_doc(valid_dataloader: DataLoader, device: torch.device, model, loss_fn, epoch: int, path: str) -> Tuple[
    float, float]:
    """"""
    total_avg_loss = 0
    total_correct = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_dataloader)):
            docs, labels, sents_per_doc, words_per_sent = batch
            docs = docs.to(device)
            labels = labels.squeeze(-1).to(device)  # B
            # print("labels:")
            # print(labels)
            logits, sent_alpha, word_alpha = model(docs, sents_per_doc.squeeze(-1).to(device), words_per_sent.to(device))
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

def train_one_epoch(train_loader, device: torch.device, model, loss_fn, optimizer, epoch, use_wandb=False):
    """
        train one epoch on specified dataloader, and do some logging.
    @param train_loader:
    @param device:
    @param model:
    @param loss_fn:
    @param optimizer:
    @param use_wandb:
    @return:
    """

    for i, batch in tqdm(enumerate(train_loader)):
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
        optimizer.zero_grad()
        loss.backward()
        # grad norm clipping prevents gradient exploding
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # calculate the accuracy
        pred_labels = torch.max(logits, dim=1)[1]
        # print(pred_labels)
        acc = torch.eq(pred_labels, labels).sum().item() / labels.shape[0]

        # print log per 1000steps
        if i % 1000 == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss_avg: {loss:.4f}\t'
                'Accuracy {acc:.3f} '.format(
                    epoch, i, len(train_loader),
                    loss=loss,
                    acc=acc
                )
            )
        if use_wandb:
            wandb.log({
                "train/loss_avg": loss,
                "train/acc": acc,
                "train/gnorm": gnorm.item()
            })


def train_one_epoch_doc(train_loader, device: torch.device, model, loss_fn, optimizer, epoch, use_wandb=False):
    """
        train one epoch on Doc dataloader, and do some logging.
    @param train_loader:
    @param device:
    @param model:
    @param loss_fn:
    @param optimizer:
    @param use_wandb:
    @return:
    """

    for i, batch in tqdm(enumerate(train_loader)):
        docs, labels, sents_per_doc, words_per_sent = batch # sents_per_doc: B x 1, words_per_sent: B x max_sent_len
        # print(sents_per_doc.shape)
        # print(words_per_sent.shape)
        docs = docs.to(device)
        labels = labels.squeeze(-1).to(device)  # B x 1 -> B
        # print("labels:")
        # print(labels)
        logits, sent_alpha, word_alpha = model(docs, sents_per_doc.squeeze(-1).to(device), words_per_sent.to(device))
        # print("logits:")
        # print(logits)
        loss = loss_fn(logits, labels)
        # print("loss")
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        # grad norm clipping prevents gradient exploding
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # calculate the accuracy
        pred_labels = torch.max(logits, dim=1)[1]  # B
        # print(pred_labels)
        acc = torch.eq(pred_labels, labels).sum().item() / labels.shape[0]

        # print log per 1000steps
        if i % 1000 == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss_avg: {loss:.4f}\t'
                'Accuracy {acc:.3f} '.format(
                    epoch, i, len(train_loader),
                    loss=loss,
                    acc=acc
                )
            )
        if use_wandb:
            wandb.log({
                "train/loss_avg": loss,
                "train/acc": acc,
                "train/gnorm": gnorm.item()
            })



def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len.to(X.device)[:, None]
    X[~mask] = value
    return X


if __name__ == '__main__':
    pass
