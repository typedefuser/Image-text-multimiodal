import torch
from models import VisionLanguageModel
from embedding import get_image_embedding, get_text_embedding
from transformers import BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionLanguageModel().to(device)
model.load_state_dict(torch.load("weights/vlm_model.pth"))
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def generate_caption(image_path):
    image_emb = torch.tensor(get_image_embedding(image_path)).to(device)
    text_input = tokenizer("Describe this image", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(image_emb, text_input)
    return tokenizer.decode(output.argmax(dim=-1))

print(generate_caption("examples/sample.jpg"))
