# Introduction 
This repository demonstrates an algorithm for finding ``good" triangulations of images.

# Algorithm

<!-- ## Vertex Locations Only -->
We start out by randomly distributing vertices over the image. Then, we iterate through the following steps:
1. Find the Delaunay triangulation of the vertices.
2. Calculate the colour of each vertex using a weighted average of the nearby pixels.
3. Reproject the vertex colours back onto the pixels via linear interpolation.
4. Calculate the squared error between the original and reprojected pixel colours.
5. Estimate the area of each vertex's Voronoi cell, and interpolate these values to the pixels.
6. Calculate the loss at each pixel by multiplying the error and interpolated area. 
7. Backpropagate the total loss value through all the above steps, and update the vertex locations to minimise it.

Minimising this loss reduces the reconstruction error over time. 
Multiplying by the vertex areas reduces the density of vertices in well-approximated areas and increases their density in poorly approximated areas.

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

