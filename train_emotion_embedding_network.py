import torch
from config import parse_configs
from emotion_embedding_network import EmotionEmbeddingNetwork
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.utils.data as tud
from dataset import IEMOCAPDataset
from dataset import EmotionEmbeddingNetworkCollate
from dataset import TacotronCollate
import numpy as np
from statistics import mean

def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def eval_accuracy(model_out, targets):
    probs = torch.softmax(model_out, dim=1)
    preds = torch.argmax(probs, dim=1)
    correct_count = float((preds==targets).sum())
    acc = float(correct_count/len(targets))
    return acc

def train(configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA", torch.cuda.is_available())
    if device.type == "cpu":
        configs.batch_size = 1

    train_data = IEMOCAPDataset(path_to_csv="data/splits/train.csv", silence=False, padded=True)
    val_data = IEMOCAPDataset(path_to_csv="data/splits/val.csv", silence=False, padded=True)

    collate_fn = EmotionEmbeddingNetworkCollate()
    train_loader = tud.DataLoader(train_data, collate_fn=collate_fn, num_workers=2, prefetch_factor=2, batch_size=configs.batch_size, shuffle=True)
    val_loader = tud.DataLoader(val_data, collate_fn=collate_fn, num_workers=2, prefetch_factor=2, batch_size=configs.batch_size, shuffle=True)

    weights = torch.from_numpy(np.load("data/weights.npy")).to(device)
    model = EmotionEmbeddingNetwork().to(device).to(torch.double)
    loss = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=.001)

    if configs.checkpoint:
        load(model, optimizer, configs.checkpoint)

    train_log_dir = f"logs/tensorboard/{configs.name}-batch_size={configs.batch_size}-lr={configs.lr}-epochs={configs.num_epochs}"
    train_summary_writer = SummaryWriter(train_log_dir)
    epochs = configs.num_epochs
    update_step = 0

    running_loss = 0.0
    running_acc = []
    for e in tqdm(range(epochs)):
        for melspecs, emotions, _ in tqdm(train_loader, leave=False):
            melspecs = melspecs.to(device).to(torch.double)
            emotions = emotions.to(device).to(torch.int64)
            model.train()
            y_pred, emotion_embedding = model(melspecs)
            J = loss(y_pred, emotions)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
            running_loss += J.item()
            running_acc += [eval_accuracy(model_out=y_pred, targets=emotions)]
            if update_step % configs.train_loss_fr == 0:
                running_loss /= configs.train_loss_fr
                running_acc = float(mean(running_acc))
                train_summary_writer.add_scalar(f'info/Training loss', running_loss, update_step)
                train_summary_writer.add_scalar(f'info/Training Accuracy', running_acc, update_step)
                running_loss = 0.0
                running_acc = []
            if update_step % configs.val_loss_fr == 0:
                val_loss = 0.0
                val_acc = []
                model.eval()
                with torch.no_grad():
                    for melspecs, emotions, _ in tqdm(val_loader, leave=False):
                        melspecs = melspecs.to(device).to(torch.double)
                        emotions = emotions.to(device).to(torch.int64)
                        y_pred = model(melspecs)[0]
                        J = loss(y_pred, emotions)
                        val_loss += J.item()
                        val_acc += [eval_accuracy(model_out=y_pred, targets=emotions)]
                val_loss /= configs.val_loss_fr
                val_acc = float(mean(val_acc))
                train_summary_writer.add_scalar(f'info/Validation loss', val_loss, update_step)
                train_summary_writer.add_scalar(f'info/Validation Accuracy', val_acc, update_step)
            update_step += 1

        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, "saved_models/" + configs.name + "_" + str(e) +"_.tar")

if __name__ == '__main__':
    configs = parse_configs()
    train(configs)


