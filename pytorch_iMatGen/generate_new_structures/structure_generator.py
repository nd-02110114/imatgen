import torch
import torch.nn as nn
from tqdm import tqdm
from os import path


from materials_generator.model import MaterialGenerator
from cell.model import CellAutoEncoder
from basis.model import BasisAutoEncoder
from utils.postprocess import image_to_cell, image_to_basis, save_basis_and_cell


class StructureGenerator(nn.Module):
    def __init__(self, device='cpu', cell_z_size=None, basis_z_size=None, z_size=None):
        super(StructureGenerator, self).__init__()
        self.device = device
        self.cell_z_size = cell_z_size
        self.basis_z_size = basis_z_size
        self.z_size = z_size
        self.cell_ae = CellAutoEncoder(z_size=cell_z_size)
        self.basis_ae = BasisAutoEncoder(z_size=basis_z_size)
        self.materials_generator = MaterialGenerator(z_size=z_size)

    def load_pretrained_weight(self, cell_ae_path, basis_ae_path, materials_generator_path):
        self.cell_ae.load_state_dict(torch.load(cell_ae_path))
        self.cell_ae.eval()
        self.basis_ae.load_state_dict(torch.load(basis_ae_path))
        self.basis_ae.eval()
        self.materials_generator.load_state_dict(torch.load(materials_generator_path))
        self.materials_generator.eval()

    def generate(self, loader, log_dir):
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                sampling_z = batch[0]
                sampling_z = sampling_z.to(self.device)

                # create image features
                out = self.materials_generator.decode(sampling_z)
                out = out.view(-1, 6, self.basis_z_size)
                # create cell image
                cell_z = out[:, 5, 0:self.cell_z_size]
                cell_image = self.cell_ae.decoder(cell_z)
                cell_image = cell_image.reshape(-1, 32, 32, 32).detach().cpu().numpy()
                # create basis image
                basis_z = out[:, 0:5, :]
                basis_image = self.basis_ae.decoder(basis_z.reshape(-1, self.basis_z_size))
                basis_image = basis_image.detach().cpu().numpy()

                # postprocess
                batch_size = cell_image.shape[0]
                # TODO
                elements = ['H', 'H', 'H', 'H', 'H']
                for i in range(batch_size):
                    try:
                        cell_atom = image_to_cell(cell_image[i])
                        basis_atoms = [
                            image_to_basis(val, elements[i])
                            for i, val in enumerate(basis_image[i*5:(i+1)*5])
                        ]
                        print(basis_atoms)
                        save_path = path.join(log_dir, 'sampling_{}.cif'.format(idx * batch_size + i))
                        save_basis_and_cell(cell_atom, basis_atoms, save_path)
                    except:  # noqa
                        pass
