from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from data import get_train_classes, load_data
from train import train, validate
from utility import save_plots
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":
    # Set random seeds 42 for reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Loading datasets
    train_loader, val_loader, test_loader = load_data()
    # Classes
    classes = ['rock', 'paper', 'scissors']
    num_classes = 3 

    #Creating the models

    # Resnet18
    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)

    # EfficientNetV2-S 
    # model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # ConvNeXt-Tiny 
    # model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    # Swin Transformer-Tiny
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)

    model_name = getattr(model, 'arch', model.__class__.__name__)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    epochs = 5

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(valid_epoch_loss)
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print("-" * 50)

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, name=model_name)
    print("TRAINING COMPLETE")

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.3f}")
    print(f"Final Test Accuracy: {test_acc:.3f}%")


    # check a single prediction
    images, labels = next(iter(test_loader))

    # Take the first sample, its papper
    img = images[0]
    label = labels[0].item()

    # De-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
    img_show = img * std + mean

    model.eval()
    with torch.no_grad():
        img_batch = img.unsqueeze(0).to(device)      # Add batch dimension
        output = model(img_batch)
        predicted_label = output.argmax(dim=1).item()
        
    classes = get_train_classes()
    true_class = classes[label]
    predicted_class = classes[predicted_label]

    print("True class:      ", true_class)
    print("Predicted class: ", predicted_class)