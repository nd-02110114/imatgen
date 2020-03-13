import math
import torch
import numpy as np
from ase.io import write
from ase import Atom, Atoms
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def post_process(x):
    x = torch.clamp(x, 0.5, 0.5001) - 0.5
    x = torch.min(x * 10000, torch.ones_like(x))
    return x


def compute_length(axis_val):
    """Calculate cell axis length"""
    non_zeros = axis_val[axis_val > 0]
    # axis_val became larger toward the center (center = 1)
    p_ijk = non_zeros.min()
    # inverse transform
    r_ijk = np.sqrt(-2 * 0.26**2 * np.log(p_ijk))
    # calculate length of the unit vector
    (a,) = np.where(axis_val == p_ijk)
    N = np.abs(16 - a[0])
    # r_ijk/r_unit = N/32
    r = r_ijk * 32.0 / float(N)
    return r


def compute_angle(ri, rj, rij):
    """Calculate angle between ri and rj"""
    # See the S3.2 in the Supplemental Information
    print(ri, rj, rij)
    cos_theta = (ri**2 + rj**2 - rij**2) / (2 * ri * rj)
    theta = math.acos(-cos_theta) * 180/np.pi  # angle in deg
    return theta


def image_to_cell(image):
    """Convert cell image to ase atom with cell information"""
    a_axis = image[:, 16, 16]
    ra = compute_length(a_axis)
    b_axis = image[16, :, 16]
    rb = compute_length(b_axis)
    c_axis = image[16, 16, :]
    rc = compute_length(c_axis)

    ab_axis = np.array([image[i, i, 16] for i in range(32)])
    rab = compute_length(ab_axis)
    bc_axis = np.array([image[16, i, i] for i in range(32)])
    rbc = compute_length(bc_axis)
    ca_axis = np.array([image[i, 16, i] for i in range(32)])
    rca = compute_length(ca_axis)

    alpha = compute_angle(rb, rc, rbc)
    beta = compute_angle(rc, ra, rca)
    gamma = compute_angle(ra, rb, rab)

    atoms = Atoms(cell=[ra, rb, rc, alpha, beta, gamma], pbc=True)
    atoms.append(Atom('Cu', [0.5]*3))
    pos = atoms.get_positions()
    atoms.set_scaled_positions(pos)
    return atoms


def detect_peaks(image):
    """Detect peaks in a basis image"""
    neighborhood = generate_binary_structure(3, 2)
    local_max = (maximum_filter(image, footprint=neighborhood, mode="wrap") == image)
    background = (image < 0.02)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = np.logical_and(local_max, np.logical_not(eroded_background))
    return detected_peaks


def image_to_basis(image, element='H'):
    """Convert basis image to ase atom with basis information"""
    # image should have dimension of (N,N,N)
    image = image.reshape(64, 64, 64)
    image0 = gaussian_filter(image, sigma=0.15)
    peaks = detect_peaks(image0)
    recon_mat = Atoms(cell=15*np.identity(3), pbc=[1, 1, 1])
    (peak_x, peak_y, peak_z) = np.where(peaks == 1.0)
    for px, py, pz in zip(peak_x, peak_y, peak_z):
        if np.sum(image[px-3:px+4, py-3:py+4, pz-3:pz+4] > 0) >= 0:
            recon_mat.append(Atom(element, (px/64.0, py/64.0, pz/64.0)))
    pos = recon_mat.get_positions()
    recon_mat.set_scaled_positions(pos)
    return recon_mat


def save_basis_and_cell(cell_atom, basis_atoms, save_path):
    """Gather basis and cell information and save cif format"""
    # gather all element
    basis_atom = basis_atoms[0]
    for atom in basis_atoms[1:]:
        basis_atom.append(atom)

    cell_pos = cell_atom.get_positions()
    pos = basis_atom.get_positions()
    delta = cell_pos - np.mean(pos, 0)
    new_pos = pos + delta

    # cell + basis
    basis_atom.set_cell(cell_atom.get_cell())
    basis_atom.set_positions(new_pos)

    write(save_path, basis_atom)
    return True
