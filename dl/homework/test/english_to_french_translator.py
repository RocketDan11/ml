import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Load the dataset
english_to_french = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle français couramment"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent des musées"),
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
    ("He sings beautifully", "Il chante magnifiquement"),
    ("They swim in the pool", "Ils nagent dans la piscine"),
    ("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
    ("She teaches English at school", "Elle enseigne l'anglais à l'école"),
    ("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
    ("He paints landscapes", "Il peint des paysages"),
    ("They laugh at the joke", "Ils rient de la blague"),
    ("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
    ("She runs in the park", "Elle court dans le parc"),
    ("We travel by train", "Nous voyageons en train"),
    ("He writes a letter", "Il écrit une lettre"),
    ("They read books at the library", "Ils lisent des livres à la bibliothèque"),
    ("The baby cries", "Le bébé pleure"),
    ("She studies hard for exams", "Elle étudie dur pour les examens"),
    ("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
    ("He fixes the car", "Il répare la voiture"),
    ("They drink coffee in the morning", "Ils boivent du café le matin"),
    ("The sun sets in the evening", "Le soleil se couche le soir"),
    ("She dances at the party", "Elle danse à la fête"),
    ("We play music at the concert", "Nous jouons de la musique au concert"),
    ("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
    ("They study French grammar", "Ils étudient la grammaire française"),
    ("The rain falls gently", "La pluie tombe doucement"),
    ("She sings a song", "Elle chante une chanson"),
    ("We watch a movie together", "Nous regardons un film ensemble"),
    ("He sleeps deeply", "Il dort profondément"),
    ("They travel to Paris", "Ils voyagent à Paris"),
    ("The children play in the park", "Les enfants jouent dans le parc"),
    ("She walks along the beach", "Elle se promène le long de la plage"),
    ("We talk on the phone", "Nous parlons au téléphone"),
    ("He waits for the bus", "Il attend le bus"),
    ("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
    ("The stars twinkle at night", "Les étoiles scintillent la nuit"),
    ("She dreams of flying", "Elle rêve de voler"),
    ("We work in the office", "Nous travaillons au bureau"),
    ("He studies history", "Il étudie l'histoire"),
    ("They listen to the radio", "Ils écoutent la radio"),
    ("The wind blows gently", "Le vent souffle doucement"),
    ("She swims in the ocean", "Elle nage dans l'océan"),
    ("We dance at the wedding", "Nous dansons au mariage"),
    ("He climbs the mountain", "Il gravit la montagne"),
    ("They hike in the forest", "Ils font de la randonnée dans la forêt"),
    ("The cat meows loudly", "Le chat miaule bruyamment"),
    ("She paints a picture", "Elle peint un tableau"),
    ("We build a sandcastle", "Nous construisons un château de sable"),
    ("He sings in the choir", "Il chante dans le chœur"),
    ("They ride bicycles", "Ils font du vélo"),
    ("The coffee is hot", "Le café est chaud"),
    ("She wears glasses", "Elle porte des lunettes"),
    ("We visit our grandparents", "Nous rendons visite à nos grands-parents"),
    ("He plays the guitar", "Il joue de la guitare"),
    ("They go shopping", "Ils font du shopping"),
    ("The teacher explains the lesson", "Le professeur explique la leçon"),
    ("She takes the train to work", "Elle prend le train pour aller au travail"),
    ("We bake cookies", "Nous faisons des biscuits"),
    ("He washes his hands", "Il se lave les mains"),
    ("They enjoy the sunset", "Ils apprécient le coucher du soleil"),
    ("The river flows calmly", "La rivière coule calmement"),
    ("She feeds the cat", "Elle nourrit le chat"),
    ("We visit the museum", "Nous visitons le musée"),
    ("He fixes his bicycle", "Il répare son vélo"),
    ("They paint the walls", "Ils peignent les murs"),
    ("The baby sleeps peacefully", "Le bébé dort paisiblement"),
    ("She ties her shoelaces", "Elle attache ses lacets"),
    ("We climb the stairs", "Nous montons les escaliers"),
    ("He shaves in the morning", "Il se rase le matin"),
    ("They set the table", "Ils mettent la table"),
    ("The airplane takes off", "L'avion décolle"),
    ("She waters the plants", "Elle arrose les plantes"),
    ("We practice yoga", "Nous pratiquons le yoga"),
    ("He turns off the light", "Il éteint la lumière"),
    ("They play video games", "Ils jouent aux jeux vidéo"),
    ("The soup smells delicious", "La soupe sent délicieusement bon"),
    ("She locks the door", "Elle ferme la porte à clé"),
    ("We enjoy a picnic", "Nous profitons d'un pique-nique"),
    ("He checks his email", "Il vérifie ses emails"),
    ("They go to the gym", "Ils vont à la salle de sport"),
    ("The moon shines brightly", "La lune brille intensément"),
    ("She catches the bus", "Elle attrape le bus"),
    ("We greet our neighbors", "Nous saluons nos voisins"),
    ("He combs his hair", "Il se peigne les cheveux"),
    ("They wave goodbye", "Ils font un signe d'adieu")
]

# Preprocess the text data
def preprocess_data():
    # Create vocabulary for English and French
    eng_words = set()
    fr_words = set()
    
    for eng, fr in english_to_french:
        for word in eng.lower().split():
            eng_words.add(word)
        for word in fr.lower().split():
            fr_words.add(word)
    
    # Add special tokens
    eng_words = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(list(eng_words))
    fr_words = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(list(fr_words))
    
    # Create dictionaries for word-to-index and index-to-word
    eng_word2idx = {word: i for i, word in enumerate(eng_words)}
    eng_idx2word = {i: word for i, word in enumerate(eng_words)}
    fr_word2idx = {word: i for i, word in enumerate(fr_words)}
    fr_idx2word = {i: word for i, word in enumerate(fr_words)}
    
    return eng_word2idx, eng_idx2word, fr_word2idx, fr_idx2word

# Define custom dataset
class TranslationDataset(Dataset):
    def __init__(self, data, eng_word2idx, fr_word2idx):
        self.data = data
        self.eng_word2idx = eng_word2idx
        self.fr_word2idx = fr_word2idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        eng_sentence, fr_sentence = self.data[idx]
        
        # Convert to indices
        eng_indices = [self.eng_word2idx.get(word.lower(), self.eng_word2idx['<unk>']) 
                       for word in eng_sentence.split()]
        fr_indices = [self.fr_word2idx['<sos>']] + \
                     [self.fr_word2idx.get(word.lower(), self.fr_word2idx['<unk>']) 
                      for word in fr_sentence.split()] + \
                     [self.fr_word2idx['<eos>']]
        
        return torch.tensor(eng_indices), torch.tensor(fr_indices)

# Encoder-Decoder model with GRU
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_length = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, target_length, target_vocab_size).to(self.device)
        
        # Get encoder outputs and hidden state
        _, encoder_hidden = self.encoder(source)
        
        # First decoder input is the <SOS> token
        decoder_input = target[:, 0].unsqueeze(1)  # Shape: [batch_size, 1]
        decoder_hidden = encoder_hidden
        
        for t in range(1, target_length):
            # Get decoder output
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Store the output
            outputs[:, t, :] = decoder_output.squeeze(1)
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token or use ground truth
            top1 = decoder_output.argmax(2)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (source, target) in enumerate(train_loader):
        source, target = source.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(source, target)
        
        # Calculate loss (ignore <SOS> token)
        loss = criterion(output[:, 1:, :].reshape(-1, output.shape[2]), target[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

# Function to evaluate the model
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (source, target) in enumerate(val_loader):
            source, target = source.to(device), target.to(device)
            
            # Forward pass
            output = model(source, target, teacher_forcing_ratio=0)
            
            # Calculate loss (ignore <SOS> token)
            loss = criterion(output[:, 1:, :].reshape(-1, output.shape[2]), target[:, 1:].reshape(-1))
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predictions = output[:, 1:, :].argmax(2)
            correct_predictions += (predictions == target[:, 1:]).sum().item()
            total_predictions += target[:, 1:].numel()
    
    return epoch_loss / len(val_loader), (correct_predictions / total_predictions) * 100

# Function to translate a sentence
def translate_sentence(sentence, model, eng_word2idx, fr_idx2word, device, vocab=None, max_length=20):
    model.eval()
    
    # Pass the vocab directly during function call
    if vocab is None:
        # Get fr_word2idx from the main function
        fr_word2idx = None
        for eng, fr in english_to_french:
            if eng == sentence:
                actual_fr = fr
                break
    else:
        fr_word2idx = vocab
    
    # Convert sentence to indices
    indices = [eng_word2idx.get(word.lower(), eng_word2idx['<unk>']) for word in sentence.split()]
    source_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    
    # Encode the sentence
    _, encoder_hidden = model.encoder(source_tensor)
    
    # Start decoding with <SOS> token
    decoder_input = torch.tensor([[1]]).to(device)  # 1 is the index for <sos>
    decoder_hidden = encoder_hidden
    
    translated_words = []
    
    for _ in range(max_length):
        # Get decoder output
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        
        # Get the highest predicted token
        predicted_token = decoder_output.argmax(2).item()
        
        # If <EOS> token, stop decoding
        if predicted_token == 2:  # 2 is the index for <eos>
            break
        
        # Add the predicted word to the translation
        if predicted_token in fr_idx2word:
            translated_words.append(fr_idx2word[predicted_token])
        
        # Update decoder input
        decoder_input = torch.tensor([[predicted_token]]).to(device)
    
    return ' '.join(translated_words)

# Main training and evaluation
def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data
    eng_word2idx, eng_idx2word, fr_word2idx, fr_idx2word = preprocess_data()
    
    # Split data into train and validation sets (80:20)
    random.shuffle(english_to_french)
    split_idx = int(0.8 * len(english_to_french))
    train_data = english_to_french[:split_idx]
    val_data = english_to_french[split_idx:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Create datasets and data loaders
    train_dataset = TranslationDataset(train_data, eng_word2idx, fr_word2idx)
    val_dataset = TranslationDataset(val_data, eng_word2idx, fr_word2idx)
    
    # Use custom collate function for padding
    def collate_fn(batch):
        eng_sentences = [item[0] for item in batch]
        fr_sentences = [item[1] for item in batch]
        
        # Pad sequences
        eng_sentences = nn.utils.rnn.pad_sequence(eng_sentences, batch_first=True, padding_value=eng_word2idx['<pad>'])
        fr_sentences = nn.utils.rnn.pad_sequence(fr_sentences, batch_first=True, padding_value=fr_word2idx['<pad>'])
        
        return eng_sentences, fr_sentences
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    input_size = len(eng_word2idx)
    output_size = len(fr_word2idx)
    hidden_size = 256
    
    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(hidden_size, output_size)
    model = EncoderDecoder(encoder, decoder, device).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=fr_word2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    
    # Lists to store losses and accuracy
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_translator_model.pt')
            print("Model saved!")
        
        print("--------------------")
    
    # Plot losses and accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('translation_training_metrics.png')
    plt.show()
    
    # Qualitative testing - test some sentences
    test_sentences = [
        "I am cold",
        "She speaks French fluently",
        "The book is on the table",
        "We celebrate birthdays with cake",
        "The sun sets in the evening"
    ]
    
    print("\nQualitative Testing:")
    for sentence in test_sentences:
        translation = translate_sentence(sentence, model, eng_word2idx, fr_idx2word, device)
        print(f"English: {sentence}")
        print(f"French (predicted): {translation}")
        
        # Find actual translation from the dataset
        for eng, fr in english_to_french:
            if eng == sentence:
                print(f"French (actual): {fr}")
                break
        print()
    
    # Final report
    print("\nFinal Training Results:")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    
    # Save vocabulary for future use
    torch.save({
        'eng_word2idx': eng_word2idx,
        'eng_idx2word': eng_idx2word,
        'fr_word2idx': fr_word2idx,
        'fr_idx2word': fr_idx2word
    }, 'translation_vocabulary.pt')

if __name__ == "__main__":
    main()