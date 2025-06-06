import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from chexnet.chexnet import DenseNet121
import gc

# RADBERT
class TextEncoder(nn.Module):
    def __init__(self, model_path="./bert/radbert_local", device='cpu'):
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
    def __init__(self, model_path="chexnet/model.pth", device='cpu'):
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
       
#Funzione di loss "leave one out" SNN
class Similarity_Loss_SNN(nn.Module):
    def __init__(self, temperature=0.05):
        super(Similarity_Loss_SNN, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings):
        B = text_embeddings.shape[0]
        total_loss = 0

        for i in range(B):
            image_i = image_embeddings[i].unsqueeze(0)
            distances_image_text = torch.norm(image_i - text_embeddings, p=2,dim=1)
            similarities_image_text = torch.exp(-distances_image_text / self.temperature)

            numerator = similarities_image_text[i]

            # Calcola il denominatore: somma delle similarità tra l'immagine i-esima
            # e tutti i testi che non sono il testo i-esimo
            denominator = torch.sum(similarities_image_text) - similarities_image_text[i]

            if denominator != 0:
                total_loss += -torch.log(numerator / denominator)
            else:
                total_loss += 0.0

        return total_loss / B


#Funzione di loss "leave one out" Sigmoid
class Similarity_Loss_Sigmoid(nn.Module):
    def __init__(self, temperature):
        super(Similarity_Loss_Sigmoid, self).__init__()
        self.temperature = temperature
        self.bias = nn.Parameter(torch.zeros(1))

# Funzione di loss "leave one out" Sigmoid (VERSIONE VETTORIZZATA)
class Similarity_Loss_Sigmoid(nn.Module):
    def __init__(self, temperature):
        super(Similarity_Loss_Sigmoid, self).__init__()
        self.temperature = temperature
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, text_embeddings, image_embeddings):
        # 1. Normalizzazione degli embedding
        normalized_image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        normalized_text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        B = text_embeddings.shape[0]
        # 2. Calcolo della matrice di similarità 
        # La moltiplicazione di matrici (dot product) tra (B, D) e (D, B) risulta in (B, B)
        # dove B è la dimensione del batch e D è la dimensione dell'embedding.
        similarity_matrix = torch.matmul(normalized_image_embeddings, normalized_text_embeddings.T)

        # 3. Applicazione del logit e del bias
        logit_matrix = (similarity_matrix * -self.temperature) + self.bias

        # 4. Creazione della matrice delle label
        # Questa matrice ha 1.0 sulla diagonale (corrispondenze positive) e -1.0 altrove (corrispondenze negative)
        labels_matrix = torch.eye(text_embeddings.shape[0], device=text_embeddings.device) * 2.0 - 1.0

        # 5. Calcolo della loss vettorizzata
        loss_matrix = torch.log(1 / (1 + torch.exp(labels_matrix * logit_matrix)))

        # 6. Somma e media della los
        total_loss = torch.sum(loss_matrix)
        
        return -(total_loss / B)
    

class Similarity_Loss_grouped_SNN(nn.Module):
    def __init__(self, temperature=0.05):
        super(Similarity_Loss_grouped_SNN, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings, groups):
        B = text_embeddings.shape[0]
        total_loss = 0

        for i in range(B):
            # Similarità tra l'immagine i-esima e TUTTI gli embedding di testo nel batch
            distances_image_text = torch.norm(image_embeddings[i].unsqueeze(0) - text_embeddings,p=2, dim=1, keepdim=True)
            similarities_image_text = torch.exp(-distances_image_text / self.temperature)

            # Numeratore: somma delle similarità con i testi dello STESSO GRUPPO escludendo se stesso
            numerator = 0
            for j in range(B):
                if groups[i] == groups[j] and i != j:
                    numerator += similarities_image_text[j]
                if numerator ==0:
                    for j in range(B):
                        if groups[i] == groups[j]:
                            numerator += similarities_image_text[j]
            # Denominatore: somma delle similarità con i testi di ALTRI GRUPPI
            denominator = 0
            for j in range(B):
                if groups[i] != groups[j]:
                    denominator += similarities_image_text[j]
                    
            if denominator != 0 and numerator != 0:
                total_loss += -torch.log(numerator / denominator)
            else:
                total_loss += 0.0

        return total_loss / B
    
