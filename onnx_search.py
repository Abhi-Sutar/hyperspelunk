import chromadb
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import config
import textwrap
import os

# Silence warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("Loading Tokenizer and ONNX Model onto RTX 3060...")

# 1. Load the tokenizer from our newly created local folder
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

# 2. Start the ONNX session and force it to use your Nvidia GPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("./onnx_model/model.onnx", providers=providers)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_collection(name=config.COLLECTION_NAME)

doc_count = collection.count()
print(f"\n--- Spelunk ONNX Search Ready! ({doc_count} pages indexed) ---")

def encode_text(text):
    """Tokenizes text, runs it through ONNX, and applies mean pooling."""
    
    # 1. Tokenize the text into numbers
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="np")
    
    # ONNX requires explicit 64-bit integers
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    # 2. Run the math through the GPU
    ort_outputs = session.run(None, ort_inputs)
    token_embeddings = ort_outputs[0] 
    attention_mask = inputs["attention_mask"]
    
    # 3. Mean Pooling (Squashing word vectors into a sentence vector)
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    
    sentence_vector = sum_embeddings / sum_mask
    return sentence_vector.tolist()

def search_index(query, top_k=3):
    # Use our new custom ONNX encoder
    query_embedding = encode_text(query)
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    if not results['documents'][0]:
        print("\nNo relevant matches found.")
        return

    print(f"\n--- Top {top_k} Results for '{query}' ---\n")
    
    matches_found = 0
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        url = results['metadatas'][0][i]['url']
        score = results['distances'][0][i]
        
        # The Quality Cutoff Threshold
        if score > 30.0:
            continue
            
        matches_found += 1
        snippet = textwrap.shorten(text, width=250, placeholder=" ... [read more]")
        
        print(f"Match {matches_found} (Distance: {score:.4f})")
        print(f"Link: {url}")
        print(f"Text: {snippet}\n")
        print("-" * 60 + "\n")
        
    if matches_found == 0:
         print("No highly relevant matches found (results were below the quality threshold).")

# --- The Interactive Loop ---
while True:
    try:
        user_query = input("\nEnter search term (or 'quit' to exit): ").strip()
        if not user_query: continue
        if user_query.lower() in ['quit', 'exit', 'q']: break
        
        search_index(user_query)
        
    except KeyboardInterrupt:
        break