"""
Demo script for the trained trait-conditioned LLM
"""

import torch
import pandas as pd
from pathlib import Path

from trait_vectorizer import TraitVectorizer
from simple_trait_token_llm import SimpleTraitTokenLLM


def load_trained_model(model_path, csv_path):
    """Load the trained model and vectorizer."""
    # Convert to strings if Path objects
    model_path = str(model_path) if hasattr(model_path, '__fspath__') else model_path
    csv_path = str(csv_path) if hasattr(csv_path, '__fspath__') else csv_path
    
    # Load vectorizer
    df = pd.read_csv(csv_path)
    vectorizer = TraitVectorizer(df)
    
    # Load model
    model = SimpleTraitTokenLLM("arnir0/Tiny-LLM", trait_dim=86)
    
    # Load trained weights
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}")
        print("Using untrained model")
    
    model.eval()
    return model, vectorizer


def generate_essay(model, vectorizer, profile_dict, essay_prompt, max_tokens=100):
    """Generate an essay for a given profile and prompt."""
    # Vectorize the profile
    trait_vector = torch.from_numpy(vectorizer.vectorize(profile_dict)).unsqueeze(0)
    
    # Create prompt
    prompt = f"{essay_prompt}\n"
    
    with torch.no_grad():
        generated = model.generate_with_traits(
            prompt=prompt,
            trait_vector=trait_vector,
            max_new_tokens=max_tokens
        )
    
    # Extract just the generated part (after the prompt)
    lines = generated.split('\n')
    if len(lines) > 1:
        return '\n'.join(lines[1:]).strip()
    else:
        return generated.split(essay_prompt)[-1].strip()


def main():
    """Demo the trained model."""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'models' / 'best_model.pt'
    csv_path = project_root / 'downloads' / 'okcupid_profiles.csv'
    
    print("Loading trained model...")
    model, vectorizer = load_trained_model(model_path, csv_path)
    
    print("\n" + "="*80)
    print("TRAIT-CONDITIONED ESSAY GENERATION DEMO")
    print("="*80)
    
    # Demo profiles with different characteristics
    demo_profiles = [
        {
            "name": "Alex (25, Software Engineer)",
            "profile": {
                'age': 25,
                'sex': 'm',
                'height': 72,
                'orientation': 'straight',
                'body_type': 'fit',
                'education': 'graduated from college/university',
                'job': 'computer / hardware / software',
                'drinks': 'socially',
                'drugs': 'never',
                'smokes': 'no',
            }
        },
        {
            "name": "Sam (30, Artist)",
            "profile": {
                'age': 30,
                'sex': 'f',
                'height': 65,
                'orientation': 'bisexual',
                'body_type': 'average',
                'education': 'working on masters program',
                'job': 'artistic / musical / writer',
                'drinks': 'rarely',
                'drugs': 'sometimes',
                'smokes': 'no',
            }
        }
    ]
    
    essay_prompts = [
        "My self summary",
        "What I'm doing with my life",
        "I'm really good at",
        "You should message me if..."
    ]
    
    # Generate essays for each profile and prompt
    for profile_info in demo_profiles:
        name = profile_info["name"]
        profile = profile_info["profile"]
        
        print(f"\n{name}")
        print("-" * len(name))
        
        for prompt in essay_prompts:
            essay = generate_essay(model, vectorizer, profile, prompt, max_tokens=50)
            print(f"\n{prompt}:")
            print(f"  {essay}")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Create your own profile and see the generated essays!")
    print("(Press Ctrl+C to exit)\n")
    
    try:
        while True:
            # Get user input
            print("Enter profile details:")
            age = int(input("Age: "))
            sex = input("Sex (m/f): ").lower()
            height = float(input("Height (inches): "))
            education = input("Education (e.g., 'college/university'): ")
            job = input("Job (e.g., 'student', 'computer / hardware / software'): ")
            
            user_profile = {
                'age': age,
                'sex': sex,
                'height': height,
                'education': education,
                'job': job,
                'orientation': 'straight',  # defaults
                'body_type': 'average',
                'drinks': 'socially',
                'drugs': 'never',
                'smokes': 'no',
            }
            
            prompt = input("\nEssay prompt: ")
            print(f"\nGenerating essay for: {prompt}")
            
            essay = generate_essay(model, vectorizer, user_profile, prompt, max_tokens=80)
            print(f"\nGenerated essay:")
            print(f"  {essay}")
            print("\n" + "-"*50)
            
    except KeyboardInterrupt:
        print("\n\nDemo complete!")


if __name__ == '__main__':
    main()