import os
import torch
from torchvision import transforms
from PIL import Image

from model import ResNet50, DenseNet121


def load_model(args, device):
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError("Unsupported model!")

    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


@torch.no_grad()
def inference(model, image_paths, device, output_file):
    transform = get_transform()
    embeddings = []

    with open(output_file, 'w') as f:
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            input_tensor = preprocess_image(img_path, transform).to(device)
            embedding = model(input_tensor).squeeze()  # shape: (embedding_dim,)
            embeddings.append(embedding.cpu())

            # Convert to string
            embedding_str = ','.join([f'{val:.6f}' for val in embedding.cpu().numpy()])
            f.write(f"{img_path}\t{embedding_str}\n")

            print(f"Inferred and saved embedding from {img_path}")

    return torch.stack(embeddings, dim=0)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Image Inference')

    parser.add_argument('--model', default='resnet50', help='Model to use')
    parser.add_argument('--embedding-dim', default=None, type=int)
    parser.add_argument('--resume', required=True, help='Path to model checkpoint')
    parser.add_argument('--image-list', nargs='+', required=True, help='List of image paths')
    parser.add_argument('--output-file', default='inference_results.txt', help='Where to save the results')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If image-list is a single directory path, expand to image paths
    image_dir = args.image_list[0]
    print(f"📁 Expanding directory: {image_dir}")
    args.image_list = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"🖼️ Found {len(args.image_list)} image files.")

    model = load_model(args, device)
    embeddings = inference(model, args.image_list, device, args.output_file)

    print(f"\n✅ Saved embeddings to: {args.output_file}")
    print(f"📐 Output embedding matrix shape: {embeddings.shape}")


if __name__ == '__main__':
    main()