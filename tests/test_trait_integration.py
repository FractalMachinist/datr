import sys
import pathlib
import torch

# Ensure the repository root is on sys.path so `src` can be imported during tests
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.simple_trait_token_llm import SimpleTraitTokenLLM


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_trait_projector_produces_gradients():
    """
    Ensure that there is a non-zero gradient from the model's next-token logits
    back to the `trait_projector` parameters. This proves the projection is
    connected in the computational graph and can be trained.
    """
    device = _get_device()

    model = SimpleTraitTokenLLM("arnir0/Tiny-LLM", trait_dim=86)
    model = model.to(device)
    model.train()

    # Create a deterministic prompt and trait vector
    prompt = "My self summary\nI am"
    trait_vector = torch.randn(1, 86, device=device)

    inputs = model.prepare_input_with_traits(prompt, trait_vector=trait_vector)
    # move tensors to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Zero grads on projector
    for p in model.trait_projector.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Forward pass: get logits for next token prediction (last position)
    outputs = model(input_ids=inputs['input_ids'], trait_vector=inputs['trait_vector'])
    # Scalar summary of logits for backward (sum ensures non-sparse gradient)
    scalar = outputs.logits[0, -1, :].sum()

    # Backprop
    scalar.backward()

    # Check trait_projector gradients
    grads = [p.grad for p in model.trait_projector.parameters()]
    assert any(g is not None and g.abs().sum().item() > 1e-8 for g in grads), (
        "Expected non-zero gradient in trait_projector parameters, but found none."
    )


def test_logits_change_for_different_traits():
    """
    Ensure that logits for the model's next-token prediction change when the
    trait vector changes. This guarantees trait conditioning affects model
    outputs.
    """
    device = _get_device()

    model = SimpleTraitTokenLLM("arnir0/Tiny-LLM", trait_dim=86)
    model = model.to(device)
    model.eval()

    prompt = "My self summary\nI am"

    trait_a = torch.randn(1, 86, device=device)
    trait_b = torch.randn(1, 86, device=device)  # different random vector

    inputs_a = model.prepare_input_with_traits(prompt, trait_vector=trait_a)
    inputs_b = model.prepare_input_with_traits(prompt, trait_vector=trait_b)

    inputs_a = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_a.items()}
    inputs_b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_b.items()}

    with torch.no_grad():
        out_a = model(input_ids=inputs_a['input_ids'], trait_vector=inputs_a['trait_vector'])
        out_b = model(input_ids=inputs_b['input_ids'], trait_vector=inputs_b['trait_vector'])

    logits_a = out_a.logits[0, -1, :].detach().cpu()
    logits_b = out_b.logits[0, -1, :].detach().cpu()

    # Assert logits are not (almost) equal
    assert not torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-4), (
        "Expected logits to differ for different trait vectors, but they were nearly equal."
    )
