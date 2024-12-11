## Dependencies to run the code:

1. install openGL packages:
  ```bash
  sudo apt-get install mesa-utils freeglut3-dev libeigen3-dev
  ```
  
2. Install or compile OpenCV (currently using 3.x, if you want to use 2.x you will need to change the #includes of opencv, otherwise it will raise some compilation errors)

## Usage

### Camera FPS controller

**'WASD'** - move the camera as in a FPS shooter game.

**'XC'** - moves the camera up and down.

**'Arrow keys'** - Perform camera rotation around itself like in a FPS shooter game.

**'ZV'** - In-plane camera rotation.

### Saving a snapshot of the simulation

Pressing **'F'** will save a snapshot of the simulation's current state. Before you do that, make sure you've created a folder called 'result' in the same folder of the binary executable. Results will be saved in the correct format that can be readily loaded by the test framework.



## Using the Auto Simulator

### Params

```bash
./nonrigid_sim_auto -h

Options:
Wind force 
[--fx <float>] [--fy <float>] [--fz <float>]
[--fx_var <float>] [--fy_var <float>] [--fz_var <float>]
Light variation, rotation and camera distance
[--light_var <float>] [--rot <float>] [--cam_dist <float>]
Out directory and Save flag
[--out_dir <string>] [--save]
Time interval to variate params and save simulation
[--variation_interval <int>] [--save_interval <int>]
```

Example
```bash
./build/nonrigid_sim_auto chambre.bmp  --fx_var 1 --fy_var 1 --fz_var 1 --fx 0 --fy 0 --fz 0 --light_var 0 --variation_interval 30 --save_interval 10 --save
```

<!-- If the recording flag is set to `0` the simulation will not capture data. This flag can be used to evaluate the deformations before the data acquisition. -->
