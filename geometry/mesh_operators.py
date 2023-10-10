import torch
from torch import Tensor


def make_differential(features: Tensor, faces: Tensor):
    """
    Given a (V, C) feature tensor and a (F, 3) faces tensor,
    returns a (F, 3, C) tensor of the signed change in feature
    values over each edge.
    """
    return features[faces[..., [[1, 2], [2, 0], [0, 1]]], :].diff(n=1, dim=-2)[..., 0, :]


def cross_product(U: Tensor, V: Tensor) -> Tensor:
    """ Calculates the batched cross-product in the last dimension of U and V."""
    return (
        U[..., [1, 2, 0]].mul(V[..., [2, 0, 1]])
        .sub(U[..., [2, 0, 1]].mul(V[..., [1, 2, 0]]))
    )


def make_mesh_geometry(edges: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes geometric properties of a triangle mesh, given tangent half-edge vectors.

    Args:
        edges (Tensor): A tensor of tangent half-edge vectors with shape (F, 3, 3).

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - the squared length of each edge with shape (F, 3),
        - the intersection areas between faces and vertex dual cells with shape (F, 3),
        - the cotangents of the angles at each corner of each face with shape (F, 3).
    """
    edge_lengths2 = edges.pow(2).sum(dim=-1)
    dots = -edges[..., [1, 2, 0], :].multiply(edges[..., [2, 0, 1], :]).sum(dim=-1)
    crosses = cross_product(-edges[..., [1, 2, 0], :], edges[..., [2, 0, 1], :]).norm(dim=-1)
    cots = dots.div(crosses)
    voronoi_areas = edge_lengths2.multiply(cots)[..., [[1, 2], [2, 0], [0, 1]]].sum(dim=-1).div(8)
    # Use 0.25-0.25-0.5 weighting for obtuse faces:
    alternative_areas = voronoi_areas.sum(dim=-1, keepdim=True).multiply(cots.lt(0).add(1).div(4))
    cell_areas = voronoi_areas.where(cots.gt(0).all(dim=-1, keepdim=True), alternative_areas)
    return (edge_lengths2, cell_areas, cots)


def sum_cell_values(cell_values: Tensor, faces: Tensor):
    """
    Aggregate the (F, C) cell_values to the vertices, output has shape (V, C).
    """
    return torch.zeros(
        (faces.max().add(1),),
        device=cell_values.device,
        dtype=cell_values.dtype,
    ).index_add(dim=0, index=faces.reshape(-1), source=cell_values.reshape(-1))

