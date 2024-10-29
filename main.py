import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from lib.u2net import U2NET


# Función para cargar el modelo preentrenado
def load_model(model_path="u2net.pth"):
    model = U2NET()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    return model


# Preprocesamiento de la imagen de entrada (sin cambiar resolución)
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Guardar el tamaño original
    transform = transforms.Compose(
        [
            transforms.Resize((320, 320)),  # Mantener 320x320 solo para el modelo
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_resized = transform(image).unsqueeze(0)  # Añadir batch
    return (
        image_resized,
        original_size,
        image,
    )  # Retorna también la imagen original para usarla luego


# Función para procesar la imagen con el modelo U2-Net
def salient_object_detection(model, image_tensor):
    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(image_tensor)
        pred = d1[:, 0, :, :]
        pred = F.interpolate(
            pred.unsqueeze(0), size=(320, 320), mode="bilinear", align_corners=False
        )
        pred = pred.squeeze().cpu().numpy()
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))  # Normalización
    return pred


# Postprocesamiento: Redimensionar la máscara a la resolución original
def apply_mask(image, mask, original_size):
    mask = cv2.resize(
        mask, original_size
    )  # Ajustar la máscara a la resolución original
    mask = np.expand_dims(mask, axis=2)
    image = np.array(image) * mask  # Aplicar la máscara a la imagen original
    # change background to white
    # convert mask to gray scale
    mask = (mask * 255).astype(np.uint8)
    y, x, _ = np.where(mask == 0)
    image[y, x] = 255
    return image, mask


# Guardar la imagen final sin el objeto saliente
def save_image(output_path, result_image):
    # conver to BGR
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR).astype(int)
    cv2.imwrite(output_path, result_image)


# Pipeline para eliminar el objeto saliente manteniendo la resolución original
def remove_salient_object(
    image_path, output_path, model_path="./models/segmentation/u2net.pth"
):
    model = load_model(model_path)
    image_tensor, original_size, original_image = preprocess_image(image_path)
    mask = salient_object_detection(model, image_tensor)
    result_image, mask_original_dim = apply_mask(original_image, mask, original_size)
    save_image(output_path, result_image)
    return mask_original_dim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove salient object from image")
    parser.add_argument(
        "--input",
        type=str,
        default="./input/baumscheibe.jpg",
        required=False,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/output.jpg",
        required=False,
        help="Path to save output image",
    )
    parser.add_argument(
        "--model", type=str, default="./input/u2net.pth", help="Path to model weights"
    )
    args = parser.parse_args()

    remove_salient_object(args.input, args.output, args.model)
