import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet

from model import ResNet50, DenseNet121

def load_distance_matrix(file_path):
    """
    Reads a distance matrix from a text file where each row 
    is space-separated distances.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # split each line into floats
    matrix = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    
    return torch.tensor(matrix, dtype=torch.float32)

def retrieval_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.to(target.device)  # move pred to the same device as target
        pred = target[pred].t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
    return res

@torch.no_grad()
def evaluate(model, loader, device, args):
    embeds, labels = [], []

    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        labels.append(_labels)

    labels = torch.cat(labels, dim=0)


    # top-k accuracy (i.e. R@K)
    #kappas = [1, 5, 10]
    kappas = [3]

    dists = load_distance_matrix("D:/DLS-X-mir/SearchAndIndexingJavaCode/dists.txt")

    # accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    # accuracy = torch.stack(accuracy).numpy()
    # print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    # mean average precision and mean precision (i.e. mAP and pr)
    # ranks = torch.argsort(dists, dim=0, descending=True)
    # mAP, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.numpy(), kappas)
    # print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    # print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))

    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = args.resume.split('/')[-1].split('.')[0]

        save_path = os.path.join(args.save_dir, file_name)
        print(labels)
        print(labels.shape)
        print("------")
        print(dists)
        print(dists.shape)
        np.savez(save_path, labels=labels.cpu().numpy(), dists=-dists.cpu().numpy())


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Choose model
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError('Model not supported!')

    if os.path.isfile(args.resume):
        print("=> loading checkpoint")
        checkpoint = torch.load(args.resume)
        if 'state-dict' in checkpoint:
            checkpoint = checkpoint['state-dict']
        model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    model.to(device)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # Set up dataset and dataloader
    if args.dataset == 'covid':
        test_dataset = ChestXrayDataSet(data_dir=args.test_dataset_dir,
                                        image_list_file=args.test_image_list,
                                        mask_dir=args.mask_dir,
                                        transform=test_transform)
    elif args.dataset == 'isic':
        test_dataset = ISICDataSet(data_dir=args.test_dataset_dir,
                                   image_list_file=args.test_image_list,
                                   mask_dir=args.mask_dir,
                                   transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)

    print('Evaluating...')
    evaluate(model, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid or isic)')
    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121 or resnet50)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./results',
                        help='Result save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
