"""ResNet-50 de torchvision: clasificador ajustado (modelo 2) y extractor congelado de embeddings (modelos 3 y 4)."""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.utils.config import CLASES, IMGSZ, MEAN, MODELOS_DIR, PIN_MEMORY, STD, SEMILLA

TF = T.Compose([T.Resize((IMGSZ, IMGSZ)), T.ToTensor(), T.Normalize(MEAN, STD)])
# aumento en linea (solo train), suave y sin volteos, ademas de las copias aumentadas que ya trae el dataset
TF_AUG = T.Compose([T.Resize((IMGSZ, IMGSZ)), T.RandomAffine(6, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                    T.ColorJitter(0.1, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
CAPAS_CONGELADAS = ("conv1", "bn1", "layer1")


class DS(Dataset):
    def __init__(self, rutas, y=None, tf=TF):
        self.rutas = list(rutas)
        self.y = np.zeros(len(self.rutas), int) if y is None else np.asarray(y)
        self.tf = tf

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        return self.tf(Image.open(self.rutas[i]).convert("RGB")), int(self.y[i])


def cargador(rutas, y=None, shuffle=False, batch=64, workers=8, aumento=False):
    g = torch.Generator()
    g.manual_seed(SEMILLA)
    return DataLoader(DS(rutas, y, TF_AUG if aumento else TF), batch_size=batch, shuffle=shuffle, num_workers=workers, generator=g,
                      pin_memory=PIN_MEMORY, persistent_workers=workers > 0)


def construir_resnet(dropout=0.3, preentrenada=True):
    pesos = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if preentrenada else None
    m = torchvision.models.resnet50(weights=pesos)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, len(CLASES)))
    for nombre, p in m.named_parameters():
        if nombre.startswith(CAPAS_CONGELADAS):
            p.requires_grad = False
    return m


def modo_entrenamiento(m):
    m.train()
    m.bn1.eval()          # las BatchNorm congeladas no actualizan sus estadisticas
    m.layer1.eval()


def extractor_congelado():
    """ResNet-50 de ImageNet sin la capa final: devuelve 2048 valores por imagen."""
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()
    return m.eval()


def cargar_resnet_ajustada(ruta=None, dev="cpu"):
    """Modelo 2 entrenado (models/resnet50.pt)."""
    ck = torch.load(ruta or MODELOS_DIR / "resnet50.pt", map_location=dev)
    assert ck["clases"] == CLASES, "El orden de clases del checkpoint no coincide"
    m = construir_resnet(ck["dropout"], preentrenada=False)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def extractor_ajustado(ruta=None):
    """La ResNet-50 del modelo 2 sin su capa final: devuelve 2048 valores por imagen."""
    m = cargar_resnet_ajustada(ruta)
    m.fc = nn.Identity()
    return m.eval()


@torch.no_grad()
def inferir(m, loader, dev, softmax=True):
    m.eval()
    out = []
    with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
        for x, _ in loader:
            z = m(x.to(dev, non_blocking=True)).float()
            out.append((torch.softmax(z, 1) if softmax else z).cpu().numpy())
    return np.vstack(out)
