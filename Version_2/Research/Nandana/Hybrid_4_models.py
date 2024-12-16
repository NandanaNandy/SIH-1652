import keras_nlp
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
from transformers import BertTokenizer, pipeline
import numpy as np
import time  # Import time module to measure execution time

# Function for Named Entity Recognition (NER)
def extract_entities(texts):
    # Load pre-trained NER pipeline from Hugging Face
    nlp_ner = pipeline("ner")
    formatted_entities = []
    
    for text in texts:
        entities = nlp_ner(text)
        formatted_entities.append(entities)  # Store raw entities for each text

    return formatted_entities

# Function to extract and format entities with confidence score
def format_entities(entities):
    formatted_output = []
    
    for entity_list in entities:
        entity_name = ""
        entity_type = ""
        confidence_score = None
        
        for entity in entity_list:
            # Combine subwords (tokens) into a single entity name if necessary
            if '##' in entity['word']:
                entity_name += entity['word'].replace('##', '')  # Remove '##' and join the parts
            else:
                if entity_name:  # Only add the previous entity when we reach a new entity
                    formatted_output.append({
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "confidence_score": f"{confidence_score*100:.1f}%"  # Format score to percentage
                    })
                entity_name = entity['word']  # Start a new entity name
                entity_type = entity['entity']
                confidence_score = entity['score']
        
        # Append the last entity
        if entity_name:
            formatted_output.append({
                "entity_type": entity_type,
                "entity_name": entity_name,
                "confidence_score": f"{confidence_score*100:.1f}%"
            })
    
    return formatted_output


# GEMMA Layer definition
class GEMMALayer(layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(GEMMALayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.embeddings = self.add_weight(
            name='gemma_embeddings',
            shape=(input_dim, self.embedding_dim),
            initializer='uniform',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.embeddings)

# Function for preprocessing unstructured text
def preprocess_text(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoded_inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="np"  # Use NumPy arrays
    )
    tokens = encoded_inputs['input_ids']
    padded_sequences = pad_sequences(tokens, padding='post')

    print("Preprocessed Tokenized Sequences:")
    print(padded_sequences)
    return padded_sequences

# Extract additional features (e.g., named entities, text length)
def extract_additional_features(texts):
    entities = extract_entities(texts)
    text_lengths = [len(text.split()) for text in texts]

    print("Extracted Named Entities:")
    formatted_entities = format_entities(entities)
    
    for entity in formatted_entities:
        print(f"Entity Type: {entity['entity_type']}, Entity Name: {entity['entity_name']}, Confidence Score: {entity['confidence_score']}")
    
    print("Extracted Text Lengths:")
    print(text_lengths)

    return formatted_entities, np.array(text_lengths).reshape(-1, 1)

# Combine textual features and additional structured features
def combine_features(text_features, additional_features):
    combined_features = np.concatenate([text_features, additional_features], axis=1)
    print("Combined Structured Features:")
    print(combined_features)
    return combined_features

# Sample input text
texts = [
    """This is to certify that Selvan Nareshkanna, son of Thiru Shanmugam, residing at Door No. 164/5, Periyar Nagar, 
    of Harur Village / Town Harur Taluk Dharmapuri District of the State of Tamil Nadu belongs to 24 Manai Telugu 
    Chetty Community, which is recognized as a Backward Class as per Government Order (Ms.) No. 85, Backward Classes,
    Most Backward Classes and Minority Welfare Department (BCC), dated 29th July 2008."""
]

# Measure execution time for preprocessing and extracting features
start_time = time.time()  # Record the start time

# Step 1: Preprocess the text (convert to tokenized sequences)
text_features = preprocess_text(texts)

# Step 2: Extract additional structured features (NER and text length)
entities, additional_features = extract_additional_features(texts)

# Step 3: Combine text features with the additional structured features
structured_features = combine_features(text_features, additional_features)

# Ensure structured_features is float32 for TensorFlow
structured_features = np.array(structured_features, dtype=np.float32)

# Example labels (binary labels for training)
y_train = np.array([0], dtype=np.float32)

# Print data shapes for debugging
print("Structured features shape:", structured_features.shape)
print("Labels shape:", y_train.shape)

# Define the model
model_input = layers.Input(shape=(structured_features.shape[1],), dtype=tf.float32)
gemma_layer = GEMMALayer(embedding_dim=128)
embedding_layer = gemma_layer(model_input)

# Reshape for LSTM input: (batch_size, timesteps, features)
reshaped_layer = layers.Reshape((embedding_layer.shape[-1], 1))(embedding_layer)

# LSTM layer
x = layers.LSTM(64, return_sequences=True)(reshaped_layer)
x = layers.GlobalAveragePooling1D()(x)

# Dense layers for classification
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

# Compile the model
model = tf.keras.Model(inputs=model_input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
model.fit(structured_features, y_train, epochs=1, batch_size=1)

# Measure execution time for the whole pipeline
end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the difference

print(f"Execution time: {execution_time:.2f} seconds")
