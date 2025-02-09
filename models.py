import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, ViTModel

class VisionLanguageModel(nn.Module):
    def __init__(self):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768 * 2, 768)  # Combining image & text features
        self.decoder = nn.Linear(768, 512)  # Output textual descriptions

    def forward(self, image, text):
        img_feat = self.vision_encoder(image).last_hidden_state[:, 0, :]
        text_feat = self.text_encoder(**text).last_hidden_state[:, 0, :]
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.decoder(self.fc(combined))

model = VisionLanguageModel()
