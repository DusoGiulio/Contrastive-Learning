import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from chexnet.chexnet import DenseNet121

# RADBERT
class TextEncoder(nn.Module):
    def __init__(self, model_path="./bert/radbert_local", device='cuda'):
        super(TextEncoder, self).__init__()
        self.device = device 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_path)
#Livelli densi aggiunti
        self.mlp1 = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),#dovrebbepassare da 768 a 1024
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.bert(**tokens, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
        return F.normalize(self.mlp2(self.mlp1(cls_embedding)), p=4, dim=1)

# Modello CheXNet per le immagini
class ImageEncoder(nn.Module):
    def __init__(self, model_path="chexnet/model.pth", device='cuda'):
        super(ImageEncoder, self).__init__()
        self.device = device 
        self.model = DenseNet121(out_size=1024)

        state_dict = torch.load(model_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model.densenet121.classifier = nn.Identity()
#Livelli densi aggiunti
        self.mlp1 = nn.Sequential(
            nn.Linear(1024, 1024),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

    def forward(self, image):
        image = image.to(self.device) 
        with torch.no_grad():
            features = self.model(image)
        return F.normalize(self.mlp2(self.mlp1(features)), p=4, dim=1)

# Modello Siamese completo
class SiameseNetwork(nn.Module):
    def __init__(self,device):  
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.text_encoder = TextEncoder(device=self.device).to(self.device)
        self.image_encoder = ImageEncoder(device=self.device).to(self.device)

    def forward(self, text, image):
        text_embedding = self.text_encoder(text)
        image_embedding = self.image_encoder(image)
        return text_embedding, image_embedding
#Funzione di loss
class Similarity_Loss_1_vs_all(nn.Module):
    def __init__(self, temperature=0.05):
        super(Similarity_Loss_1_vs_all, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings, positive_pairs):
        B = text_embeddings.shape[0]
        total_loss = 0

        for i in range(B):
            text_i = text_embeddings[i].unsqueeze(0)
            distances = torch.norm(text_i - image_embeddings, dim=1)
            similarities = torch.exp(-distances / self.temperature)

            numerator = similarities[i]
            denominator = torch.sum(similarities) - similarities[i]

            total_loss += -torch.log(numerator / denominator)

        return total_loss / B