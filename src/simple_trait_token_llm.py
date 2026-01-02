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


class SimpleTraitTokenLLM(nn.Module):
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
        
        # Get embedding layer (use model-agnostic accessor)
        self.embedding_layer = self.model.get_input_embeddings()
        self.hidden_dim = self.model.config.hidden_size
        self.traits_token_id = self.tokenizer.convert_tokens_to_ids('[TRAITS]')
        
        print(f"Model: {model_name}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Vocab size: {len(self.tokenizer)}")
        print(f"[TRAITS] token ID: {self.traits_token_id}")
        
        # Trait projection layer
        self.trait_projector = nn.Linear(trait_dim, self.hidden_dim)
        # Initialize trait projector weights for stable projections
        nn.init.xavier_uniform_(self.trait_projector.weight)
        
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

    def _build_inputs_embeds(self, input_ids: torch.Tensor, trait_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build `inputs_embeds` from `input_ids` and (optionally) inject trait-conditioned
        embeddings at positions of the `[TRAITS]` token.

        This single helper centralizes the logic so `forward` and `generate_with_traits`
        both reuse the same behavior and there is no duplication.

        Args:
            input_ids: (batch, seq_len)
            trait_vector: Optional (batch, trait_dim) or (trait_dim,)

        Returns:
            inputs_embeds: (batch, seq_len, hidden_dim)
        """
        # Base token embeddings
        inputs_embeds = self.embedding_layer(input_ids)

        if trait_vector is None:
            return inputs_embeds

        # Find [TRAITS] positions
        trait_positions = (input_ids == self.traits_token_id)
        if not trait_positions.any():
            return inputs_embeds

        # Normalize trait vector shape to (batch, trait_dim)
        trait_tensor = trait_vector
        if trait_tensor.dim() == 1:
            trait_tensor = trait_tensor.unsqueeze(0)

        # Broadcast single trait vector to batch if needed
        if trait_tensor.size(0) != input_ids.size(0):
            if trait_tensor.size(0) == 1:
                trait_tensor = trait_tensor.repeat(input_ids.size(0), 1)
            else:
                raise ValueError("Batch size of trait_vector does not match input_ids")

        trait_tensor = trait_tensor.to(inputs_embeds.device)
        trait_embeddings = self._get_trait_embedding(trait_tensor)  # (batch, hidden_dim)

        # Inject trait embeddings for every trait token position in the batch
        for b in range(input_ids.size(0)):
            mask = trait_positions[b]
            if mask.any():
                inputs_embeds[b, mask] = trait_embeddings[b]

        return inputs_embeds
    
    def forward(self, input_ids, attention_mask=None, trait_vector=None, **kwargs):
        """
        Forward pass with optional trait conditioning.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask
            trait_vector: Optional trait vector (batch_size, trait_dim)
            **kwargs: Other model arguments
        """
        # If no trait vector, use normal forward pass (faster path)
        if trait_vector is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

        # Build inputs_embeds with trait injection
        inputs_embeds = self._build_inputs_embeds(input_ids, trait_vector)

        # Forward pass with modified embeddings
        return self.model(
            inputs_embeds=inputs_embeds,
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
                           max_new_tokens: int = 50,
                             do_sample: bool = True,
                             temperature: float = 1.0,
                             top_k: int = 50,
                             top_p: float = 0.95,
                             num_return_sequences: int = 1,
                             **kwargs):
        """
        Generate text conditioned on traits.

        This implementation builds `inputs_embeds` from the model embedding
        layer, replaces embeddings at the special `[TRAITS]` token positions
        with trait-conditioned embeddings, then calls `self.model.generate`
        so the trait information is included in cached `past_key_values` and
        thus influences the entire generation (not just the initial context).

        Args:
            prompt: Text prompt
            trait_vector: Trait vector (batch_size, trait_dim) or (trait_dim,)
            max_new_tokens: Number of tokens to generate
            do_sample: Whether to sample (stochastic) or use greedy/beam
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Top-p (nucleus) sampling
            num_return_sequences: Number of samples to return per prompt
            **kwargs: Additional generate kwargs passed through

        Returns:
            Generated string (or list of strings if `num_return_sequences>1`).
        """
        # Prepare input
        inputs = self.prepare_input_with_traits(prompt, trait_vector)

        # Move tensors to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        # Build inputs_embeds using centralized helper so `forward` and generation
        # behavior is identical and there's no duplicated logic.
        with torch.no_grad():
            inputs_embeds = self._build_inputs_embeds(input_ids, inputs.get('trait_vector'))

            gen_kwargs = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Merge any user-provided kwargs, letting explicit args override
            gen_kwargs.update(kwargs)

            generated_ids = self.model.generate(**gen_kwargs)

            # generated_ids shape: (batch_size * num_return_sequences, seq_len)
            generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

            if len(generated_texts) == 1:
                return generated_texts[0]
            return generated_texts


def test_simple_trait_model():
    """Test the simple trait-conditioned model."""
    print("="*80)
    print("TESTING SIMPLE TRAIT-CONDITIONED LLM")
    print("="*80)
    
    # Create model
    model = SimpleTraitTokenLLM("arnir0/Tiny-LLM", trait_dim=86)
    
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
    starting_point = "I am"
    generated = model.generate_with_traits(
        prompt="My self summary\n" + starting_point,
        trait_vector=trait_vector,
        max_new_tokens=20
    )
    print(f"  Generated: {starting_point} {generated}")
    
    return model


if __name__ == '__main__':
    model = test_simple_trait_model()
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print("\nSimple trait-conditioned LLM is working!")
    print("Ready to train on the OkCupid dataset.")