import scipy.spatial
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def make_faces(verts: Tensor):
    """
    Input is a (V, D) float tensor.
    Returns a (F, D + 1) long tensor of the Delaunay triangualation.
    """
    return torch.from_numpy(
        scipy.spatial.Delaunay(
            verts.cpu().detach().double().numpy(),
            qhull_options="i Qt Qbb Qc Qz Q12",
        ).simplices
    ).to(device=verts.device, dtype=torch.long)


class ScreenTriangulation(nn.Module):
    def __init__(self, V: int = 1_024, D: int = 3):
        super().__init__()
        (self.V, self.D) = (V, D)
        self.verts_logit = Parameter(torch.rand(self.V, 2).logit())
        self.verts_features = Parameter(torch.rand((self.V, self.D)))
        self.register_buffer(
            'boundary',
            torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        )

    def forward(self,):
        verts_2d = self.verts_logit.sigmoid().mul(2).sub(1)
        verts_2d[:4] = self.boundary.clone()
        verts_3d = F.pad(verts_2d, [0, 1], "constant", 1.0)
        faces = make_faces(verts_2d)
        verts_features_packed = self.verts_features
        return (verts_3d, faces, verts_features_packed)
