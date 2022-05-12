import torch
from config import parse_configs
from classifier import EmbedNet
from tensorboardX import SummaryWriter
from tqdm import tqdm

def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train(configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA", torch.cuda.is_available())
    if device.type == "cpu":
        configs.batch_size = 1
    # TODO: add dataloaders
    train_loader = []
    val_loader = []

    # TODO: change path
    weights = torch.load("../iemocap/weights.pt")
    model = EmbedNet().to(device)
    loss = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=.001)

    if configs.checkpoint:
        load(model, optimizer, configs.checkpoint)
    model.to(torch.double)


    train_log_dir = 'logs/tensorboard/' + configs.name
    train_summary_writer = SummaryWriter(train_log_dir)
    epochs = configs.num_epochs
    update_step = 0

    running_loss = 0.0
    for e in tqdm(range(epochs)):
        for X, y in tqdm(train_loader, leave=False):
            model.train()
            y_pred = model(X)
            J = loss(y_pred, y)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
            running_loss += J.item()
            if update_step % configs.train_loss_fr == 0:
                running_loss /= configs.train_loss_fr
                train_summary_writer.add_scalar(f'info/Training loss', running_loss, update_step)
                running_loss = 0.0
            if update_step % configs.val_loss_fr == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for X, y in tqdm(val_loader, leave=False):
                        y_pred = model(X)
                        J = loss(y_pred, y)
                        val_loss += J.item()
                train_summary_writer.add_scalar(f'info/Validation loss', val_loss, update_step)
            update_step += 1

        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, "saved_models/" + configs.name + "_" + str(e) +"_.tar")

if __name__ == '__main__':
    configs = parse_configs()
    train(configs)


