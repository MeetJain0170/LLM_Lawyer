    """
Complete Training Pipeline for Legal LLM
Includes: Data Loading, Training Loop, Validation, Checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from pathlib import Path
import time
import math
from tqdm import tqdm

class LegalTextDataset(Dataset):
    """
    Dataset for legal text training
    Loads pre-tokenized data from JSONL files
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(data['text'])
                
        print(f"Loaded {len(self.examples)} examples")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            # Pad with PAD token
            pad_id = self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]
            token_ids = token_ids + [pad_id] * (self.max_length - len(token_ids))
            
        # Convert to tensor
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)  # Input
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)  # Target (shifted by 1)
        
        return input_ids, target_ids

class LegalLLMTrainer:
    """
    Complete training system for the legal LLM
    """
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        config: dict
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'] * len(self.train_loader),
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log every N steps
            if self.global_step % self.config['log_interval'] == 0:
                self.log_metrics({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': self.current_epoch,
                    'step': self.global_step
                })
                
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for input_ids, target_ids in tqdm(self.val_loader, desc="Validating"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, loss = self.model(input_ids, target_ids)
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {path}")
        
    def log_metrics(self, metrics: dict):
        """Log training metrics"""
        # Write to file
        log_file = self.checkpoint_dir / "training_log.jsonl"
        with open(log_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')
            
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config['num_epochs']}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Total steps: {self.config['num_epochs'] * len(self.train_loader)}")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, perplexity = self.validate()
            
            print(f"\nEpoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(f"New best model! Val loss: {val_loss:.4f}")
                
            # Log epoch metrics
            self.log_metrics({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'perplexity': perplexity,
                'best_val_loss': self.best_val_loss
            })
            
        print("Training complete!")

# Example usage
if __name__ == "__main__":
    from legal_llm_model import LegalLLM  # Import the model we created
    from bpe_tokenizer import BPETokenizer  # Import tokenizer
    
    # Configuration
    config = {
        'batch_size': 8,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'grad_clip': 1.0,
        'log_interval': 100,
        'checkpoint_dir': 'checkpoints',
        'max_seq_len': 512
    }
    
    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.load("legal_tokenizer.pkl")
    
    # Create datasets
    train_dataset = LegalTextDataset(
        "data/train.jsonl",
        tokenizer,
        max_length=config['max_seq_len']
    )
    
    val_dataset = LegalTextDataset(
        "data/val.jsonl",
        tokenizer,
        max_length=config['max_seq_len']
    )
    
    # Create model
    model = LegalLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=config['max_seq_len']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = LegalLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    trainer.train()