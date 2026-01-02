"""
Training Script for Trait-Conditioned LLM

Brings together:
- TraitVectorizer (trait encoding)
- OkCupidDataset (data loading)
- SimpleTraitConditionedLLM (model)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import wandb
from tqdm import tqdm

from trait_vectorizer import TraitVectorizer
from dataset_loader import OkCupidDataset
from simple_trait_llm import SimpleTraitConditionedLLM


class TrainingConfig:
    """Training configuration."""
    def __init__(self):
        self.model_name = "arnir0/Tiny-LLM"
        self.trait_dim = 86
        self.max_length = 256
        self.batch_size = 8
        self.learning_rate = 5e-4
        self.num_epochs = 5
        self.warmup_steps = 100
        self.save_steps = 500
        self.eval_steps = 200
        self.logging_steps = 50
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        
        # Dataset
        self.max_essays = None # Go big or go home
        self.train_split = 0.8
        
        # Paths
        self.data_dir = Path(__file__).parent.parent
        self.csv_path = self.data_dir / "downloads" / "okcupid_profiles.csv"
        self.output_dir = self.data_dir / "models"
        self.logs_dir = self.data_dir / "logs"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


def collate_fn(batch, tokenizer, max_length=256):
    """
    Custom collate function for trait-conditioned training.
    
    Args:
        batch: List of (trait_vector, essay_text) tuples
        tokenizer: Model tokenizer
        max_length: Max sequence length
    """
    trait_vectors = []
    texts = []
    
    for trait_vector, essay_text in batch:
        trait_vectors.append(torch.from_numpy(trait_vector))
        texts.append(f"[TRAITS] {essay_text}")
    
    # Stack trait vectors
    trait_vectors = torch.stack(trait_vectors)  # (batch_size, trait_dim)
    
    # Tokenize texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        'trait_vectors': trait_vectors,
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': encoded['input_ids'].clone()  # For causal LM, labels = input_ids
    }


def train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            trait_vector=batch['trait_vectors'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item()
        num_batches += 1
        
        if step % config.logging_steps == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}'
            })
            
            # Log to wandb if available
            if wandb.run:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/learning_rate': lr,
                    'train/step': epoch * len(dataloader) + step
                })
        
        # Save checkpoint
        if step % config.save_steps == 0 and step > 0:
            checkpoint_path = config.output_dir / f"checkpoint-epoch{epoch}-step{step}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'step': step,
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                trait_vector=batch['trait_vectors'],
                labels=batch['labels']
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training function."""
    # Config
    config = TrainingConfig()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb (skip for testing)
    # wandb.init(
    #     project="trait-conditioned-llm",
    #     config=vars(config),
    #     dir=str(config.logs_dir)
    # )
    
    print("Loading dataset...")
    # Load data and create vectorizer
    df = pd.read_csv(config.csv_path)
    vectorizer = TraitVectorizer(df)
    
    # Create dataset
    dataset = OkCupidDataset(
        str(config.csv_path),
        vectorizer,
        max_essays=config.max_essays
    )
    
    # Split into train/val
    n = len(dataset)
    train_size = int(n * config.train_split)
    val_size = n - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Load model
    print("Loading model...")
    model = SimpleTraitConditionedLLM(config.model_name, config.trait_dim)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ONLY TRAIN THE TRAIT PROJECTOR
    # Freeze all model parameters except the trait projector
    for param in model.parameters():
        param.requires_grad = False
    for param in model.trait_projector.parameters():
        param.requires_grad = True
    
    # Show trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create data loaders
    def collate_wrapper(batch):
        return collate_fn(batch, model.tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=2
    )
    
    # Optimizer and scheduler
    # Only pass the trait projector parameters to optimizer
    optimizer = optim.AdamW(
        model.trait_projector.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_steps
    )
    
    print(f"Training for {config.num_epochs} epochs ({total_steps} steps)")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, device, epoch)
        
        # Evaluate
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'val/loss': val_loss,
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = config.output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model: {best_model_path}")
    
    # Save final model
    final_model_path = config.output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    
    # Create a test trait vector
    test_profile = {
        'age': 25,
        'sex': 'f',
        'height': 65,
        'orientation': 'straight',
        'body_type': 'average',
        'drinks': 'socially'
    }
    
    test_trait_vector = torch.from_numpy(vectorizer.vectorize(test_profile)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate_with_traits(
            prompt="My self summary\nI am",
            trait_vector=test_trait_vector,
            max_new_tokens=30
        )
        print(f"Generated text: {generated}")
    
    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()