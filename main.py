from collections import defaultdict
from pathlib import Path
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from tqdm import tqdm

from geometry.fragments_operators import (
    average_pixel_features_to_verts,
    blend_face_layers,
    interpolate_vertex_features,
)
from geometry.mesh_operators import (
    make_differential,
    make_mesh_geometry,
    sum_cell_values,
)
from models.triangulation import ScreenTriangulation

# Set Model Parameters:
## Mesh Parameters:
num_vertices = 4_096
optimise_colours = False

## Fitting parameters:
device = "cuda"
steps = 2_001
learning_rate = 4e-3
blur_radius = 0
faces_per_pixel = 1
sigma2 = 1e-5

## Output Options:
vis_steps = 4  # display and write every vis_steps frames
frame_rate = 24.0

input_images = sorted(glob.glob('data/input/*'))
for image_path in input_images:
    image_name = image_path.split('.')[-2].split('/')[-1]
    output_dir = Path(f'data/output/{image_name}')

    # Read image:
    pixel_features = (
        torchvision.io.read_image(image_path).to(device)
        .add(1/2).div(256).movedim(-3, -1)[None, ..., :3]
    )

    # Set Derived Parameters:
    (_, H, W, _) = pixel_features.shape
    num_features = (3 if optimise_colours else 0)
    triangulation = ScreenTriangulation(V=num_vertices, D=num_features).to(device)
    print(f"Input image shape: {tuple(pixel_features.shape[1:])}. \nMesh shape: ({num_vertices}, 2).")

    rasterizer = MeshRasterizer(
        cameras=FoVOrthographicCameras(
            znear=0.5, zfar=2.0,
            max_y=1.0, min_y=-1.0,
            max_x=1.0, min_x=-1.0,
            scale_xyz=((max([W / H, 1]), max([H / W, 1]), 1),),
            R=torch.eye(3)[None],
            T=torch.zeros((3,))[None],
            K=None,
            device="cuda",
        ),
        raster_settings=RasterizationSettings(
            image_size=(H, W),
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(triangulation.parameters(), lr=learning_rate)

    # Initialise visualisation windows and video writers:
    windows = ['Interpolated', 'Area', 'Mesh']
    Path.mkdir(output_dir, exist_ok=True)
    video_writers = {}
    for window in windows:
        cv2.startWindowThread()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, W, H)
        video_writers[window] = cv2.VideoWriter(
            filename=str(output_dir / f'{window}.mp4'), 
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
            fps=frame_rate,
            frameSize=(W, H)
        )

    stats_log = defaultdict(list)

    # Optimise triangulation:
    loop = tqdm(range(steps))
    for step in loop:
        optimizer.zero_grad()

        ## Create mesh and rasterize:
        (verts_3d, faces, verts_features_packed) = triangulation()
        meshes = Meshes(verts_3d[None], faces[None])
        fragments = rasterizer(meshes)
        verts, faces = (meshes.verts_packed(), meshes.faces_packed())

        ## Calculate reprojection error:
        if optimise_colours:
            vertex_features = verts_features_packed
        else:
            vertex_features = average_pixel_features_to_verts(
                faces_packed=faces,
                pix_to_face=fragments.pix_to_face,
                bary_coords=fragments.bary_coords,
                pixel_features=pixel_features,
            )
        interpolated_features = interpolate_vertex_features(
            vertex_features=vertex_features,
            faces=faces,
            pix_to_face=fragments.pix_to_face,
            bary_coords=fragments.bary_coords,
        )
        interpolated_features = blend_face_layers(
            layer_pixel_features=interpolated_features,
            pix_to_face=fragments.pix_to_face,
            dists=fragments.dists,
            sigma=sigma2,
        )
        error = pixel_features.sub(interpolated_features).pow(2)

        ## Calculate vertex density and cotangent values:
        edges = make_differential(verts, faces)
        (edge_lengths2, cell_areas, cots) = make_mesh_geometry(edges)
        vertex_areas = sum_cell_values(cell_values=cell_areas, faces=faces)
        interpolated_vertex_area = interpolate_vertex_features(
            vertex_features=vertex_areas[..., None],
            faces=faces,
            pix_to_face=fragments.pix_to_face,
            bary_coords=fragments.bary_coords,
        )
        interpolated_vertex_area = blend_face_layers(
            layer_pixel_features=interpolated_vertex_area,
            pix_to_face=fragments.pix_to_face,
            dists=fragments.dists,
            sigma=sigma2,
        )

        ## Calculate loss and back-propagate:
        loss = (
            error.multiply(interpolated_vertex_area).mean().log()
            # + cots.pow(2).mean().log().div(2)
        )
        loss.backward()
        loop.set_description_str(str(error.mean().pow(1/2).item()))
        optimizer.step()

        stats = {
            'RMS Error': error.mean().pow(1/2).item(),
        }
        for stat in stats:
            stats_log[stat].append(stats[stat])

        # Visualise frames and save to video:
        if (step%vis_steps==0)or(step==steps-1):
            visualisations = {
                'Interpolated': (
                    interpolated_features
                    .mul(255).add(1/2).clamp(0, 255).to(torch.uint8)[0, ..., [2, 1, 0]]
                ),
                'Area':(
                    interpolated_vertex_area
                    .div(interpolated_vertex_area.max())  # normalise
                    .pow(1/2.33)  # gamma correction
                    .mul(255).add(1/2).clamp(0, 255).to(torch.uint8)[0, ..., [0, 0, 0]]
                ),
                'Mesh': (
                    fragments.dists
                    .abs().div(-sigma2).exp()
                    .sub(1).mul(-1)  # flip colours
                    .mul(255).add(1/2).clamp(0, 255).to(torch.uint8)[0, ..., [0, 0, 0]]
                )
            }
            for window in windows:
                vis = visualisations[window].cpu().numpy()
                video_writers[window].write(vis)
                cv2.imshow(window, vis)
            
            if step in {0, 16, 80, 400, 1200, 2000}:
                print(interpolated_features.shape, fragments.dists.shape)
                torchvision.utils.save_image(
                    (
                        fragments.dists
                        .abs().div(-sigma2).exp()
                        .sub(1).mul(-1)
                    ).movedim(-1, 0), 
                    output_dir / f'Mesh_{step}.png'
                )
                torchvision.utils.save_image(
                    (
                        interpolated_features
                    ).movedim(-1, -3), 
                    output_dir / f'Interpolated_{step}.png'
                )
                torchvision.utils.save_image(
                    (
                        interpolated_vertex_area
                        .div(interpolated_vertex_area.max())  # normalise
                        .pow(1/2.33)  # gamma correction 
                    ).movedim(-1, -3), 
                    output_dir / f'Area_{step}.png'
                )
            

            cv2.waitKey(1)

    # Save plots:
    for stat in stats_log:
        fig = pd.Series(stats_log[stat]).plot().get_figure()
        plt.xlabel('Step')
        plt.ylabel(stat)
        fig.savefig(output_dir / f'{stat}.png')
        plt.close()

    cv2.destroyAllWindows()
