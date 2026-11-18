from backend.services.model_manager import load_hf_model

models = ["gpt2", "distilgpt2"]

for model_name in models:

    print("\nTesting:", model_name)

    loaded = load_hf_model(model_name)

    print("Tokenizer:", loaded["tokenizer"].__class__.__name__)
    print("Model:", loaded["model"].__class__.__name__)
    print("Device:", loaded["device"])

print("\nBoth Hugging Face models loaded successfully!")