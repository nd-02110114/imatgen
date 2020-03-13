import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from os import path, makedirs, getcwd


from utils.seed import seed_everything
from structure_generator import StructureGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training VAE for materials generator')
    # for data
    parser.add_argument('--materials-generator-path', default='../materials_generator/result/best_model.pth',
                        type=str, help='path to materials generator model (relative path)')
    parser.add_argument('--cell-ae-path', default='../cell/result/best_model.pth',
                        type=str, help='path to cell autoencoder path (relative path)')
    parser.add_argument('--basis-ae-path', default='../basis/result/best_model.pth',
                        type=str, help='path to basis autoencoder path (relative path)')
    parser.add_argument('--out-dir', '-o', default='result',
                        type=str, help='path for output directory')

    # for model
    parser.add_argument('--cell-z-size', default=20, type=int,
                        help='size for latent variable (200) in cell image auto encoder')
    parser.add_argument('--basis-z-size', default=200, type=int,
                        help='size for latent variable (500) in basis image auto encoder')
    parser.add_argument('--z-size', default=500, type=int,
                        help='size for latent variable (200) in materials generator')

    parser.add_argument('--gpu', '-g', action='store_true', help='using gpu during training')
    parser.add_argument('--sampling-size', default=10000, type=int, help='using gpu during training')
    parser.add_argument('--sampling', choices=['random', 'slerp'], default='random',
                        help='choose sampling method, (default: random)')
    parser.add_argument('--batch-size', '-b', default=8, type=int,
                        help='mini-batch size (8)')
    parser.add_argument('--seed', default=1234, type=int, help='seed value (default: 1234)')

    return parser.parse_args()


def load_model(model, device, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def main():
    # get args
    args = parse_arguments()

    # make output directory
    out_dir = args.out_dir
    out_dir_path = path.normpath(path.join(getcwd(), out_dir))
    makedirs(out_dir_path, exist_ok=True)
    # save the parameter
    with open(path.join(out_dir_path, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    # setup
    seed_everything(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    materials_generator_path = path.normpath(path.join(getcwd(), args.materials_generator_path))
    cell_ae_path = path.normpath(path.join(getcwd(), args.cell_ae_path))
    basis_ae_path = path.normpath(path.join(getcwd(), args.basis_ae_path))

    # sampling
    if args.sampling == 'random':
        size = 500 * args.sampling_size
        sampling = np.random.normal(size=size).reshape((args.sampling_size, 500))
    else:
        size = 500 * args.sampling_size
        sampling = np.random.normal(size=size).reshape((args.sampling_size, 500))

    # create dataset
    dataset = TensorDataset(torch.FloatTensor(sampling))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # generate
    structure_generator = StructureGenerator(device, args.cell_z_size, args.basis_z_size, args.z_size)
    structure_generator.load_pretrained_weight(cell_ae_path, basis_ae_path, materials_generator_path)
    structure_generator.generate(loader, out_dir_path)


if __name__ == '__main__':
    main()
