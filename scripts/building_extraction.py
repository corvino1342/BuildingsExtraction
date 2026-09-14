import argparse
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Ignora warning su file non georiferiti
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Importa il modello (adatta il percorso)
from src.models.unet import UNetLL

### Funzioni di supporto da `grid_tiling.py` e `infer_probability_maps.py`

def scale_meters_pixels(bbox: Tuple[float, float, float, float], image: Image.Image) -> Tuple[float, float]:
    """Converte metri in pixel per un bbox dato."""
    min_lat, min_lon, max_lat, max_lon = bbox
    image_width, image_height = image.size
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians((min_lat + max_lat) / 2))
    scale_lat = (max_lat - min_lat) * meters_per_degree_lat / image_height
    scale_lon = (max_lon - min_lon) * meters_per_degree_lon / image_width
    return scale_lat, scale_lon

def generate_grid_in_pixels(bbox: Tuple[float, float, float, float], image_width: int, image_height: int, tile_size_meters: float) -> List[Dict]:
    """Genera una griglia di tile in coordinate pixel."""
    scale_lat, scale_lon = scale_meters_pixels(bbox, Image.new("RGB", (image_width, image_height)))
    tile_size_pixels_y = int(tile_size_meters / scale_lat)
    tile_size_pixels_x = int(tile_size_meters / scale_lon)
    tiles = []
    for y in range(0, image_height, tile_size_pixels_y):
        for x in range(0, image_width, tile_size_pixels_x):
            tile = {
                "bbox_pixel": [(y, x), (y, x + tile_size_pixels_x), (y + tile_size_pixels_y, x + tile_size_pixels_x), (y + tile_size_pixels_y, x)],
                "bbox_real": [
                    (bbox[0] + (y / image_height) * (bbox[2] - bbox[0]), bbox[1] + (x / image_width) * (bbox[3] - bbox[1])),
                    (bbox[0] + (y / image_height) * (bbox[2] - bbox[0]), bbox[1] + ((x + tile_size_pixels_x) / image_width) * (bbox[3] - bbox[1])),
                    (bbox[0] + ((y + tile_size_pixels_y) / image_height) * (bbox[2] - bbox[0]), bbox[1] + ((x + tile_size_pixels_x) / image_width) * (bbox[3] - bbox[1])),
                    (bbox[0] + ((y + tile_size_pixels_y) / image_height) * (bbox[2] - bbox[0]), bbox[1] + (x / image_width) * (bbox[3] - bbox[1]))
                ]
            }
            tiles.append(tile)
    return tiles

def get_bbox_from_json(image_path: str, metadata_path: str = "bbox_metadata.json") -> Tuple[float, float, float, float]:
    """Legge il bbox da un file JSON in base al nome dell'immagine."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    image_name = os.path.basename(image_path)
    if image_name not in metadata:
        raise ValueError(f"Bbox non trovato per {image_name} in {metadata_path}")
    return tuple(metadata[image_name])

def load_model(model_path: str, device: torch.device) -> UNetLL:
    """Carica il modello UNetLL."""
    model = UNetLL(n_channels=3, n_classes=1)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(model: UNetLL, image_tensor: torch.Tensor) -> np.ndarray:
    """Esegue inferenza e restituisce la mappa di probabilità."""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs.astype(np.float32)


### Funzione principale

def main():
    parser = argparse.ArgumentParser(description="Estrai edifici da una mappa usando inferenza per tile.")
    parser.add_argument("--input", type=str, required=True, help="Percorso dell'immagine di input (es. mappa.png).")
    parser.add_argument("--model", type=str, required=True, help="Percorso del modello addestrato (es. model.pth).")
    parser.add_argument("--output_dir", type=str, required=True, help="Cartella di output per i risultati.")
    parser.add_argument("--metadata_path", type=str, default="bbox_metadata.json", help="Percorso del file JSON con i bbox (default: bbox_metadata.json).")
    parser.add_argument("--tile_size_meters", type=float, default=30, help="Dimensione del tile in metri (default: 30).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soglia per la binarizzazione (default: 0.5).")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo per l'inferenza (cuda/cpu).")
    args = parser.parse_args()

    # Leggi il bbox dal JSON
    bbox = get_bbox_from_json(args.input, args.metadata_path)

    # Crea la cartella di output
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Carica l'immagine
    image = Image.open(args.input).convert("RGB")
    image_width, image_height = image.size

    # Genera la griglia di tile
    tiles = generate_grid_in_pixels(args.bbox, image_width, image_height, args.tile_size_meters)

    # Carica il modello
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = load_model(args.model, device)

    # Transform per normalizzare l'immagine (adatta ai tuoi dati)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Elabora ogni tile
    for i, tile in enumerate(tiles):
        y1, x1 = tile["bbox_pixel"][0]
        y2, x2 = tile["bbox_pixel"][2]
        tile_img = image.crop((x1, y1, x2, y2))
        tile_tensor = transform(tile_img).unsqueeze(0)

        # Esegue inferenza
        prob_map = predict(model, tile_tensor)

        # Salva la mappa di probabilità per il tile
        tile_output_dir = Path(args.output_dir) / f"tile_{i}"
        tile_output_dir.mkdir(exist_ok=True)
        np.save(tile_output_dir / "prob_map.npy", prob_map)

        # Binarizza la mappa (opzionale)
        binary_map = (prob_map > args.threshold).astype(np.uint8)
        Image.fromarray(binary_map * 255).save(tile_output_dir / "binary_mask.png")

        # Salva i metadati del tile
        with open(tile_output_dir / "metadata.json", "w") as f:
            json.dump({
                "bbox_pixel": tile["bbox_pixel"],
                "bbox_real": tile["bbox_real"],
                "threshold": args.threshold
            }, f, indent=2)

    print(f"Elaborazione completata. Risultati salvati in {args.output_dir}")

if __name__ == "__main__":
    main()