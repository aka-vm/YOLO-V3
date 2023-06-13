import torch
import torch.optim as optim

from tqdm import tqdm

import utils
import config
from yolo3 import YOLOv3
from loss import YoloLoss
from dataset import YOLODataset

# Enable env vars
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# def test(model):

def train_epoch(
    model,
    train_dataloader,
    optimizer,
    loss_fxn,
    anchor_boxes,
    scaler = None,
):
    """
    Parameters:
        model: the model to train
        train_dataloader: the dataloader for training
        val_dataloader[optional]: the dataloader for validation
        optimizer: the optimizer to use
        anchor_boxes: the anchor boxes to use
        scaler[optional]: the scaler to use for mixed precision training

    """
    loop = tqdm(train_dataloader, leave=True)
    batch_losses = []
    for batch_idx, (images, target_bboxes) in enumerate(loop):
        images = images.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            output_bboxes = model(images)
            batch_loss = sum([
                loss_fxn(out, target_bboxes[i].to(config.DEVICE), anchor_boxes[i])
                for i, out in enumerate(output_bboxes)
            ])

        batch_losses.append(batch_loss)
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = torch.mean(torch.stack(batch_losses))
        loop.set_postfix(loss=mean_loss.item())

    return mean_loss.item()


def train_model(
    model: torch.nn.Module,
    train_dataloader,
    epochs: int,
    optimizer,
    loss_fxn,
    val_dataloader = None,
    scaler = None,
    anchor_boxes = None,
    **kwargs
):
    """
    Parameters:
        model: the model to train
        train_dataloader: the dataloader for training
        val_dataloader[optional]: the dataloader for validation
        epochs: number of epochs to train
        optimizer: the optimizer to use
        scaler[optional]: the scaler to use for mixed precision training
        anchor_boxes: the anchor boxes to use

    """
    assert scaler, "Might wanna use scaler for now; use torch.cuda.amp.GradScaler()"

    # the train forward of the model
    # Constants
    anchor_boxes = anchor_boxes or torch.tensor(config.ANCHORS)
    anchor_boxes = (anchor_boxes * \
        torch.reshape(torch.tensor(config.S), (3, 1, 1)).repeat(1, 3, 2)).to(config.DEVICE)

    train_metrics = {"losses": [], "class_accuracy": [], "mAP": []}
    val_metrics = {"losses": [], "class_accuracy": [], "mAP": []}
    for epoch in range(epochs):
        epoch_loss = train_epoch(
            model, train_dataloader, optimizer, loss_fxn, anchor_boxes, scaler
        )
        train_metrics["losses"].append(epoch_loss)


        # if val_dataloader:
        #     # Create a test function
        #     # loss, class_accuracy, mAP

        #     val_metrics["losses"]
        #     val_metrics["class_accuracy"]
        #     val_metrics["mAP"]

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss}")


def main():
    train_csv_path = config.DATASET_PATH / "20examples.csv"
    train_dataloader, test_dataloader, val_dataloader = utils.get_dataloaders(train_csv_path, train_val_split=0).values()
    # train_dataloader, test_dataloader, val_dataloader = utils.get_dataloaders(train_val_split=0.15).values()

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fxn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = config.NUM_EPOCHS

    train_model(
        model,
        train_dataloader,
        num_epochs,
        optimizer,
        loss_fxn,
        val_dataloader,
        scaler = scaler,
    )

    # test model
    # class_accuracy, mAP

if __name__ == "__main__":
    main()