from rnn_model import load_model, generate_text

def main():
    # Load the trained model
    model, char_to_idx, idx_to_char = load_model('rnn_model.pth')
    
    # Read some seed text from input-text.txt
    with open('input-text.txt', 'r') as f:
        text = f.read()
        seed_text = text[:50]  # Use first 50 characters as seed
    
    # Generate text with different temperatures
    temperatures = [0.5, 0.8, 1.0]
    
    print("Original seed text:")
    print("-" * 50)
    print(seed_text)
    print("-" * 50)
    print()
    
    for temp in temperatures:
        print(f"\nGenerated text (temperature={temp}):")
        print("-" * 50)
        generated = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=seed_text,
            predict_len=200,  # Generate 200 characters
            temperature=temp
        )
        print(generated)
        print("-" * 50)

if __name__ == "__main__":
    main()
