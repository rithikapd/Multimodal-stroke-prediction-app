import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalStrokeModel(nn.Module):
    def __init__(self):
        super(MultiModalStrokeModel, self).__init__()

        self.mri_model = models.resnet50(weights=None)
        self.ct_model = models.resnet50(weights=None)

        self.mri_model.fc = nn.Identity()
        self.ct_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, mri, ct):

        mri_feat = self.mri_model(mri)
        ct_feat = self.ct_model(ct)

        combined = torch.cat((mri_feat, ct_feat), dim=1)

        out = self.classifier(combined)

        return out

model = MultiModalStrokeModel()

model.load_state_dict(
    torch.load("models/final_multimodal_stroke_model.pth", map_location=device)
)

model.to(device)
model.eval()


# -------------------------------
# Image Transform
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -------------------------------
# Prediction Function
# -------------------------------

def predict_stroke(mri_path, ct_path):

    mri = Image.open(mri_path).convert("RGB")
    ct = Image.open(ct_path).convert("RGB")

    mri = transform(mri).unsqueeze(0).to(device)
    ct = transform(ct).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(mri, ct)

        prob = torch.softmax(output, dim=1)

    confidence = prob.max().item()
    pred = prob.argmax().item()

    label = "Stroke Detected" if pred == 1 else "Normal"

    return label, round(confidence * 100, 2)