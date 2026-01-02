"""
Simple Trait-Conditioned LLM Implementation

Instead of complex hooks, we'll inject traits by:
1. Adding special [TRAITS] tokens to the input
2. Replacing their embeddings with trait-conditioned embeddings
3. This way we work with the model as-is without modifying internals
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class SimpleTraitConditionedLLM(nn.Module):
    """
    Trait-conditioned LLM using special trait tokens.
    
    Strategy:
    - Add [TRAITS] token to vocabulary
    - Replace its embedding with trait-conditioned vector
    - Prepend to all inputs: "[TRAITS] My self summary\n..."
    """
    
    def __init__(self, model_name: str, trait_dim: int = 86):
        super().__init__()
        
        self.trait_dim = trait_dim
        self.model_name = model_name
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens
        self.tokenizer.add_tokens(['[TRAITS]'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Get embedding layer
        self.embedding_layer = self.model.model.embed_tokens
        self.hidden_dim = self.model.config.hidden_size
        self.traits_token_id = self.tokenizer.convert_tokens_to_ids('[TRAITS]')
        
        print(f"Model: {model_name}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Vocab size: {len(self.tokenizer)}")
        print(f"[TRAITS] token ID: {self.traits_token_id}")
        
        # Trait projection layer
        self.trait_projector = nn.Linear(trait_dim, self.hidden_dim)
        
        # Initialize trait embedding to zero (preserve original behavior)
        with torch.no_grad():
            self.embedding_layer.weight[self.traits_token_id].fill_(0)
    
    def _get_trait_embedding(self, trait_vector: torch.Tensor) -> torch.Tensor:
        """
        Convert trait vector to embedding space.
        
        Args:
            trait_vector: (batch_size, trait_dim)
            
        Returns:
            trait_embedding: (batch_size, hidden_dim)
        """
        return self.trait_projector(trait_vector)
    
    def forward(self, input_ids, attention_mask=None, trait_vector=None, **kwargs):
        """
        Forward pass with optional trait conditioning.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask
            trait_vector: Optional trait vector (batch_size, trait_dim)
            **kwargs: Other model arguments
        """
        # If no trait vector, use normal forward pass
        if trait_vector is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Get input embeddings
        input_embeddings = self.embedding_layer(input_ids)  # (batch, seq_len, hidden_dim)
        
        # Find positions of [TRAITS] tokens
        trait_positions = (input_ids == self.traits_token_id)
        
        if trait_positions.any():
            # Get trait embeddings
            trait_embeddings = self._get_trait_embedding(trait_vector)  # (batch, hidden_dim)
            
            # Replace [TRAITS] token embeddings with trait-conditioned embeddings
            # We need to broadcast trait_embeddings to all trait positions
            for batch_idx in range(input_ids.size(0)):
                trait_mask = trait_positions[batch_idx]
                if trait_mask.any():
                    # Replace all [TRAITS] positions with the same trait embedding
                    input_embeddings[batch_idx, trait_mask] = trait_embeddings[batch_idx]
        
        # Forward pass with modified embeddings
        return self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def prepare_input_with_traits(self, text: str, trait_vector: Optional[torch.Tensor] = None) -> dict:
        """
        Prepare input text with trait token prepended.
        
        Args:
            text: Input text (e.g., "My self summary\nI am a...")
            trait_vector: Optional trait vector
            
        Returns:
            dict with input_ids, attention_mask, trait_vector
        """
        # Prepend trait token to text
        if trait_vector is not None:
            text_with_traits = f"[TRAITS] {text}"
        else:
            text_with_traits = text
        
        # Tokenize
        inputs = self.tokenizer(
            text_with_traits,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        inputs['trait_vector'] = trait_vector
        return inputs
    
    def generate_with_traits(self, prompt: str, trait_vector: torch.Tensor, 
                           max_new_tokens: int = 50, **kwargs):
        """
        Generate text conditioned on traits.
        
        Args:
            prompt: Text prompt
            trait_vector: Trait vector (1, trait_dim)
            max_new_tokens: Number of tokens to generate
            **kwargs: Generation parameters
        """
        # Prepare input
        inputs = self.prepare_input_with_traits(prompt, trait_vector)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate (note: this uses the original generate method, so trait conditioning
        # only applies to the initial context, not the generated tokens)
        with torch.no_grad():
            input_ids = inputs['input_ids']
            
            # Get initial embeddings with traits
            initial_output = self.forward(
                input_ids=input_ids,
                trait_vector=inputs['trait_vector']
            )
            
            # For simplicity, use greedy decoding from the logits
            generated_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Get next token logits
                outputs = self.forward(
                    input_ids=generated_ids,
                    trait_vector=inputs['trait_vector']
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            return generated_text


def test_simple_trait_model():
    """Test the simple trait-conditioned model."""
    print("="*80)
    print("TESTING SIMPLE TRAIT-CONDITIONED LLM")
    print("="*80)
    
    # Create model
    model = SimpleTraitConditionedLLM("arnir0/Tiny-LLM", trait_dim=86)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy trait vector
    trait_vector = torch.randn(1, 86, device=device)
    
    print(f"\nTrait vector shape: {trait_vector.shape}")
    
    # Test 1: Forward pass without traits
    print("\n1. Forward pass WITHOUT traits:")
    prompt = "My self summary\nI am a creative person who"
    inputs = model.prepare_input_with_traits(prompt, trait_vector=None)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'])
    
    print(f"  Input: {prompt}")
    print(f"  Input tokens: {model.tokenizer.decode(inputs['input_ids'][0])}")
    print(f"  Output shape: {outputs.logits.shape}")
    
    # Test 2: Forward pass with traits
    print("\n2. Forward pass WITH traits:")
    inputs_with_traits = model.prepare_input_with_traits(prompt, trait_vector=trait_vector)
    inputs_with_traits = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs_with_traits.items()}
    
    with torch.no_grad():
        outputs_with_traits = model(**inputs_with_traits)
    
    print(f"  Input with traits: {model.tokenizer.decode(inputs_with_traits['input_ids'][0])}")
    print(f"  Output shape: {outputs_with_traits.logits.shape}")
    
    # Test 3: Generation
    print("\n3. Text generation with traits:")
    generated = model.generate_with_traits(
        prompt="My self summary\nI am",
        trait_vector=trait_vector,
        max_new_tokens=20
    )
    print(f"  Generated: {generated}")
    
    # Check parameter counts
    print("\n4. Parameter analysis:")
    total_params = sum(p.numel() for p in model.parameters())
    trait_params = sum(p.numel() for p in model.trait_projector.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trait projection parameters: {trait_params:,}")
    print(f"  Trait params as % of total: {100*trait_params/total_params:.3f}%")
    
    return model


if __name__ == '__main__':
    model = test_simple_trait_model()
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print("\nSimple trait-conditioned LLM is working!")
    print("Ready to train on the OkCupid dataset.")