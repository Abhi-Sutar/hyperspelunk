from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

print("Downloading and converting PyTorch model to ONNX...")
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Setting export=True forces Optimum to do the conversion on the fly
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Saving ONNX model to local folder...")
model.save_pretrained("./onnx_model")
tokenizer.save_pretrained("./onnx_model")

print("Export 100% Complete!")
