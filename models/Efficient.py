import torch
import torch.nn as nn
from torch.nn import functional as F
import time


class EFFICIENT_B0:
    @staticmethod
    def fine_tune(model, train_loader, val_loader, model_config):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
        model.to(model_config.device)
        optimizer = model_config.optimizer_fn(model)
        best_val_acc = 0.0

        for epoch in range(model_config.num_epochs):
            start_time = time.time()

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(model_config.device)
                labels = labels.to(model_config.device)

                outputs = model(images)
                loss = model_config.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = 100 * correct / total
            end_train = time.time()

            print(
                f"Epoch [{epoch+1}/{model_config.num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"| Train Time: {(end_train - start_time):.2f}s"
            )

            # Validation
            val_acc, val_loss, val_time = EFFICIENT_B0.evaluate(
                model, val_loader, device=model_config.device, silent=True
            )
            print(
                f"Validation Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} "
                f"| Val Time: {val_time:.2f}s"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = f"{model_config.out_name}_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✔️ Saved best model to {checkpoint_path} with acc {val_acc:.2f}%")

        print(f"🎯 Training finished. Best Val Accuracy: {best_val_acc:.2f}%")

    @staticmethod
    def evaluate(model, test_loader, device="cpu", silent=False):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        loss_total = 0

        start_time = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time = time.time()
        acc = 100 * correct / total
        avg_loss = loss_total / len(test_loader.dataset)
        eval_time = end_time - start_time

        if not silent:
            print(f"Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f} | Time: {eval_time:.2f}s")

        return acc, avg_loss, eval_time
