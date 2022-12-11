# Naive SDF generation in Python

![Figure](./Figure.png)

Very naive SDF implementation inspired by https://prideout.net/blog/distance_fields/, without the hull construction.

Converts any image to BW, and thresholds it at value 128.  
Displays the contour and outputs the SDF as a uint8 PNG.


Example: `$ python sdf_gen.py world.png`
<p float="left">
  <img src="./world.png" width="32%" />
  <img src="./Figure2.png" width="32%" /> 
  <img src="./world-sdf.png" width="32%" /> 
</p>
