def log_message(message):
    print(f"[LOG] {message}")

def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    log_message(f"Model checkpoint saved to {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    log_message(f"Model checkpoint loaded from {filepath}")
    return epoch, loss

def visualize_images(images, nrow=8, title=None):
    grid = make_grid(images, nrow=nrow)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()