import torch
from torch.nn import functional as F
import os


class VisionTransfomers:
    @staticmethod
    def fine_tune(model, train_loader, val_loader, model_config):
        model.to(model_config.device)
        optimizer = model_config.optimizer_fn(model)
        scheduler = model_config.scheduler(optimizer=optimizer)
        best_val_acc = 0.0

        for epoch in range(model_config.num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(model_config.device)
                labels = labels.to(model_config.device)

                outputs = model(images)
                logits = outputs.logits  # ✅ Extract logits for loss & accuracy

                loss = model_config.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(logits, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            print(
                f"Epoch [{epoch+1}/{model_config.num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )
            scheduler.step()

            # ----- Validation -----
            val_acc, val_loss = VisionTransfomers.evaluate(model, val_loader, model_config.device)
            print(f"Validation Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = f"{model_config.out_name}_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved with acc {val_acc:.2f}% to {checkpoint_path}")

        print(f"Training completed. Best Val Accuracy: {best_val_acc:.2f}%")

    @staticmethod
    def evaluate(model, test_loader, device="cpu"):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        loss_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # loss = F.cross_entropy(outputs, labels)
                logits = outputs.logits  # ✅ Extract logits

                loss = F.cross_entropy(logits, labels)
                loss_total += loss.item()
                # _, predicted = torch.max(outputs.data, 1)
                _, predicted = torch.max(logits, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = loss_total / len(test_loader)

        print(f"Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f}")
        return acc, avg_loss