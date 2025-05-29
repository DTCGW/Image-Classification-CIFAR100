import torch
from transformers import SwinForImageClassification
from config import SwinConfig

def build_model():
    return SwinForImageClassification.from_pretrained(
        SwinConfig.model_name,
        num_labels=100,
        ignore_mismatched_sizes=True,
    )

def train_model(model, train_loader, test_loader, config):
    model.to(config.device)
    optimizer = config.optimizer_fn(model)
    best_acc = 0.0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = config.criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"[SwinTransformer] Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Accuracy={acc:.2f}%")

        # Evaluation after each epoch
        val_acc, val_loss = evaluate(model, test_loader, config)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config.out_name}_best.pt")
            print(f"[SwinTransformer] Best model saved with accuracy {best_acc:.2f}%")

def evaluate(model, test_loader, config):
    model.to(config.device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = config.criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"[SwinTransformer] Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f}")
    return acc, avg_loss
