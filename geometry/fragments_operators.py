import torch
from torch import Tensor


def interpolate_vertex_features(
    vertex_features: Tensor,
    faces: Tensor,
    pix_to_face: Tensor,
    bary_coords: Tensor
):
    """
    vertex_features: (V, C) float
    faces: (F, 3) long
    pix_to_face: (N, H, W, P) long
    bary_coords: (N, H, W, P, 3) float

    returns: (N, H, W, P, C) float
    """
    return (
        vertex_features[faces, :][pix_to_face, :, :]
        .multiply(bary_coords[..., None])
        .sum(dim=-2)
    )

def average_pixel_features_to_verts(
    faces_packed: Tensor,
    pix_to_face: Tensor,
    bary_coords: Tensor,
    pixel_features: Tensor,
):
    """
    faces: (F, 3) long
    pix_to_face: (N, H, W, P) long
    bary_coords: (N, H, W, P, 3) float
    pixel_features: (N, H, W, C)

    returns: (N, H, W, P, C) float
    """
    bary_coords = bary_coords.where(bary_coords.ne(-1), torch.zeros_like(bary_coords))
    pix_to_verts = faces_packed[pix_to_face, :]
    buffer = torch.zeros(
        (faces_packed.max().add(1), pixel_features.size(-1)),
        device=pixel_features.device,
    )
    idx = pix_to_verts[..., None].reshape(-1)
    numer_src = (
        pixel_features[..., None, None, :]
        .multiply(bary_coords[..., None])
        .reshape(-1, pixel_features.size(-1))
    )
    numer = buffer.index_add(dim=0, index=idx, source=numer_src)
    denom_src = (
        torch.ones_like(pixel_features[..., None, None, :])
        .multiply(bary_coords[..., None])
        .reshape(-1, pixel_features.size(-1))
    )
    denom = buffer.index_add(dim=0, index=idx, source=denom_src)
    return numer.div(denom)


def blend_face_layers(
    layer_pixel_features: Tensor,
    pix_to_face: Tensor,
    dists: Tensor,
    sigma: float = 1e-4,
):
    """
    pixel_features: (N, H, W, P, C) float
    pix_to_face: (N, H, W, P) long
    dists: (N, H, W, P) float
    sigma: float

    returns: (N, H, W, C) float
    """
    mask = pix_to_face.ne(-1)
    weights = dists.div(-sigma).sigmoid().multiply(mask)
    denom = weights[..., None].sum(dim=-2)
    denom = denom.where(denom.ne(0), torch.ones_like(denom))
    result = layer_pixel_features.mul(weights[..., None]).sum(dim=-2).div(denom)
    return result
