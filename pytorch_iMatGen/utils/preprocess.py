import numpy as np
from ase import Atoms
from joblib import Parallel, delayed

from utils.constant import COD_ATOMLIST, MP_ATOMLIST


def get_fakeatoms_positioning_in_the_grid(atoms, nbins):
    """Create dummy atoms positioning in the grid."""
    # fakeatoms for grid
    grid = np.array([i/float(nbins) for i in range(nbins)])
    yv, xv, zv = np.meshgrid(grid, grid, grid)
    pos = np.zeros((nbins**3, 3))
    pos[:, 0] = xv.flatten()
    pos[:, 1] = yv.flatten()
    pos[:, 2] = zv.flatten()
    # making pseudo-crystal containing H positioned at pre-defined fractional coordinate
    fakeatoms_grid = Atoms('H' + str(nbins**3))
    fakeatoms_grid.set_cell(atoms.get_cell())
    fakeatoms_grid.set_pbc(True)
    fakeatoms_grid.set_scaled_positions(pos)
    return fakeatoms_grid


def get_scale(sigma):
    """Get a scale for images"""
    scale = 1.0 / (2 * sigma ** 2)
    return scale


def get_image_one_atom(atom, fakeatoms_grid, nbins):
    """Create one image from one ase atom object."""
    grid_copy = fakeatoms_grid.copy()
    image = np.zeros((1, nbins**3))
    grid_copy.append(atom)
    drijk = grid_copy.get_distances(-1, range(0, nbins**3), mic=True)
    scale = get_scale(sigma=0.26)
    pijk = np.exp(-scale * drijk ** 2)
    image[:, :] = pijk.flatten()
    return image.reshape(nbins, nbins, nbins)


def get_all_atomlabel(all_atomlist=None):
    """Get an element information."""
    if all_atomlist is None:
        all_atomlist = list(set(MP_ATOMLIST + COD_ATOMLIST))

    all_atomlist = sorted(all_atomlist)
    return all_atomlist


def ase_atoms_to_image(ase_atoms, nbins, all_atomlist, num_cores):
    """Create images from ase atom objects. (multi process)"""
    fakeatoms_grid = get_fakeatoms_positioning_in_the_grid(ase_atoms, nbins)
    # so slow...
    imageall_gen = Parallel(n_jobs=num_cores)(
        delayed(get_image_one_atom)(atom, fakeatoms_grid, nbins) for atom in ase_atoms)
    imageall_list = list(imageall_gen)
    all_atomlist = get_all_atomlabel(all_atomlist)

    channellist = []
    for i, atom in enumerate(ase_atoms):
        channellist.append(atom.symbol)

    channellist = sorted(list(set(channellist)))
    nc = len(channellist)
    image = np.zeros((nbins, nbins, nbins, nc))
    for i, atom in enumerate(ase_atoms):
        nnc = channellist.index(atom.symbol)
        img_i = imageall_list[i]
        image[:, :, :, nnc] += img_i * (img_i >= 0.02)

    return image, channellist


def basis_translate(ase_atoms):
    """Create dummy atoms for basis images."""
    N = len(ase_atoms)
    pos = ase_atoms.positions
    cg = np.mean(pos, 0)
    dr = 7.5 - cg  # move to center of 15A-cubic box
    dpos = np.repeat(dr.reshape(1, 3), N, 0)
    new_pos = dpos + pos
    atoms_ = ase_atoms.copy()
    atoms_.cell = 15.0 * np.identity(3)
    atoms_.positions = new_pos
    return atoms_


def cell_translate(ase_atoms):
    """Create dummy atoms for cell images."""
    cell = ase_atoms.cell
    atoms_ = Atoms('H')
    atoms_.cell = cell
    atoms_.set_scaled_positions([0.5, 0.5, 0.5])
    return atoms_
