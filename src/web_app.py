"""
Flask Web App for Trait-Conditioned Essay Generation
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import torch
import pandas as pd
from pathlib import Path

from trait_vectorizer import TraitVectorizer
from simple_trait_token_llm import SimpleTraitTokenLLM



app = Flask(__name__, static_folder="../public", static_url_path="/")
CORS(app)

# Global variables for model and vectorizer
model = None
vectorizer = None


def load_model():
    """Load the trained model and vectorizer."""
    global model, vectorizer
    
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'models' / 'checkpoint-epoch0-step10000.pt'
    csv_path = project_root / 'downloads' / 'okcupid_profiles.csv'
    
    print("Loading model...")
    # Load vectorizer
    df = pd.read_csv(str(csv_path))
    vectorizer = TraitVectorizer(df)
    
    # Load model
    model = SimpleTraitTokenLLM("arnir0/Tiny-LLM", trait_dim=86)

    # Load trained weights if available
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location='cpu')

        # checkpoint can be either a raw state_dict or a dict containing 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Try to load state dict with non-strict mode to tolerate wrapper/prefix mismatches
        try:
            load_result = model.load_state_dict(state_dict, strict=False)
            # load_state_dict returns an _IncompatibleKeys object when strict=False
            try:
                missing = getattr(load_result, 'missing_keys', None)
                unexpected = getattr(load_result, 'unexpected_keys', None)
                print(f"Loaded trained weights from {model_path} (strict=False)")
                if missing:
                    print(f"Missing keys when loading checkpoint: {missing}")
                if unexpected:
                    print(f"Unexpected keys in checkpoint: {unexpected}")
            except Exception:
                # Some torch versions return None; ignore
                pass
        except RuntimeError as e:
            # Provide a helpful error path: try nested 'model' key if present
            print(f"Initial load_state_dict failed: {e}")
            if isinstance(state_dict, dict) and 'model' in state_dict:
                try:
                    load_result = model.load_state_dict(state_dict['model'], strict=False)
                    print(f"Loaded trained weights from nested 'model' key in {model_path} (strict=False)")
                except Exception as e2:
                    print(f"Nested model load failed: {e2}")
                    raise
            else:
                raise
    else:
        print(f"Warning: No trained weights found at {model_path}")
        print("Using untrained model")
    
    model.eval()
    print("Model loaded successfully!")



# Serve index.html for root
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')



# All essay prompts from README.md
ALL_ESSAY_PROMPTS = [
    "My self summary",
    "What I’m doing with my life",
    "I’m really good at",
    "The first thing people usually notice about me",
    "Favorite books, movies, show, music, and food",
    "The six things I could never do without",
    "I spend a lot of time thinking about",
    "On a typical Friday night I am",
    "The most private thing I am willing to admit",
    "You should message me if..."
]

# Stream essays one at a time
@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    profile_data = request.get_json()
    def essay_stream():
        try:
            if model is None or vectorizer is None:
                import json
                yield json.dumps({'prompt': 'Error', 'text': 'Model not loaded'}) + '\n'
                return
            trait_vector = torch.from_numpy(vectorizer.vectorize(profile_data)).unsqueeze(0)
            for prompt in ALL_ESSAY_PROMPTS:
                with torch.no_grad():
                    generated = model.generate_with_traits(
                        prompt=f"{prompt}\n",
                        trait_vector=trait_vector,
                        max_new_tokens=50
                    )
                    lines = generated.split('\n')
                    if len(lines) > 1:
                        essay_text = '\n'.join(lines[1:]).strip()
                    else:
                        essay_text = generated.split(prompt)[-1].strip()
                    essay_text = essay_text.replace('[TRAITS]', '').strip()
                    import json
                    yield json.dumps({'prompt': prompt, 'text': essay_text}) + '\n'
        except Exception as e:
            import json
            yield json.dumps({'prompt': 'Error', 'text': str(e)}) + '\n'
    return Response(essay_stream(), mimetype='text/plain')


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🚀 Starting Datr Web App")
    print("="*60)
    print("Visit: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)