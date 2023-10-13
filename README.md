# Introduction 
This repository demonstrates an algorithm for finding ``good" triangulations of images. In this context, a triangulation is good if it generates a piecewise-linear approximation of the target image with low reconstruction error.

# Algorithm
<!-- ## Vertex Locations Only -->
The image data consists of 1) the locations and 2) colour values for each pixel, which are stored in a $HW\times 2$ matrix and a $HW\times 3$ matrix, respectively. We start out by randomly distributing vertices over the image storing their positions in a $V\times2$ real matrix. We haven't calculated vertex colours yet, but can note that they will occupy a $V\times3$ matrix.
$$
\texttt{pixels}: \mathbb{R}^{HW\times 2} \\
\texttt{pixel\_colours}: \mathbb{R}^{HW\times 3} \\
\texttt{verts}: \mathbb{R}^{V\times 2} \\
\texttt{vertex\_colours}: \mathbb{R}^{V\times 3} 
$$
Then, we iterate through the following steps:
1. Find the Delaunay triangulation of the vertices.
2. Calculate the colour of each vertex using a weighted average of the nearby pixels.
$$
\texttt{aggregate}: (\texttt{verts},\texttt{ pixels},\texttt{ pixel\_colours})\mapsto\texttt{vertex\_colours} 
$$
3. Reproject the vertex colours back onto the pixels via linear interpolation.
$$
\texttt{interpolate}: (\texttt{verts},\texttt{ pixels},\texttt{ vertex\_colours})\mapsto\texttt{pixel\_colours} \\
\texttt{reprojection} = \texttt{interpolate}(\texttt{verts},\texttt{ pixels},\texttt{ aggregate(\texttt{verts},\texttt{ pixels},\texttt{ pixel\_colours})}) 
$$
4. Calculate the error between the original and reprojected pixel colours.
$$
\texttt{error} = \texttt{reprojection} - \texttt{pixel\_colours} 
$$
5. Estimate the area of each vertex's Voronoi cell, and interpolate these values to the pixels.
$$
\texttt{vertex\_areas} : \texttt{verts} \to \mathbb{R}^V \\
\texttt{pixel\_areas} = \texttt{interpolate}(\texttt{verts},\texttt{ pixels},\texttt{ vertex\_areas}(\texttt{verts})) 
$$
6. Calculate the loss at each pixel by multiplying the error and interpolated area. 
$$
\texttt{loss} = \texttt{error}^2 \times \texttt{pixel\_areas} 
$$
7. Backpropagate the total loss value through all the above steps, and update the vertex locations to minimise it.
$$
\texttt{total\_loss} = \sum\texttt{loss} 
$$

Minimising this loss reduces the reconstruction error over time. 
Multiplying by the vertex areas reduces the density of vertices in well-approximated areas, and increases their density in poorly approximated areas.

# Gallery
From left to right, the images below the mesh, the reconstructed image, and the interpolated vertex areas.
<p align="center">
  <img src="images/Mesh_0.png?raw=true" width="250">
  <img src="images/Interpolated_0.png?raw=true" width="250">
  <img src="images/Area_0.png?raw=true" width="250">
</p>
<p align="center">
  <img src="images/Mesh_16.png?raw=true" width="250">
  <img src="images/Interpolated_16.png?raw=true" width="250">
  <img src="images/Area_16.png?raw=true" width="250">
</p>
<p align="center">
  <img src="images/Mesh_80.png?raw=true" width="250"> 
  <img src="images/Interpolated_80.png?raw=true" width="250"> 
  <img src="images/Area_80.png?raw=true" width="250">
</p>
</p>
<p align="center">
  <img src="images/Mesh_400.png?raw=true" width="250"> 
  <img src="images/Interpolated_400.png?raw=true" width="250"> 
  <img src="images/Area_400.png?raw=true" width="250">
</p>
</p>
<p align="center">
  <img src="images/Mesh_1200.png?raw=true" width="250"> 
  <img src="images/Interpolated_1200.png?raw=true" width="250"> 
  <img src="images/Area_1200.png?raw=true" width="250">
</p>
<p align="center">
  <img src="images/Mesh_2000.png?raw=true" width="250">
  <img src="images/Interpolated_2000.png?raw=true" width="250">
  <img src="images/Area_2000.png?raw=true" width="250">
</p>
<p align="center">
  <img src="data/output/test1/RMS%20Error.png" width="600">
</p>

<!-- ## Vertex Locations and Colour -->

# References:
1. Optimal Delaunay Triangulations, https://www.math.uci.edu/~chenlong/Papers/Chen.L%3BXu.J2004.pdf.

