## Getting started

Start by cloning the repo:
```bash
git clone https://github.com/YadiraF/PIXIE
cd PIXIE
```  

#### Requirements
  * Python 3.7 (numpy, skimage, scipy, opencv, kornia)  
  * PyTorch >= 1.6 
    You can run 
    ```bash
    pip install -r requirements.txt
    ```
      Or create a separate virtual environment by running:  
    ```bash
    bash install_conda.sh
    ```
    or 
    ```bash
    bash install_pip.sh
    ```
    For visualization, we use our [rasterizer](https://github.com/YadiraF/PIXIE/tree/master/pixielib/utils/rasterizer) that uses pytorch JIT Compiling Extensions. 
    If there occurs a compiling error, you can install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) instead and set --rasterizer_type=pytorch3d when running the demos. 

#### Pre-trained model and data
  * Download [SMPL-X Model](http://smpl-x.is.tue.mpg.de/). Choose 'SMPL-X 2020' to download 'SMPLX_NEUTRAL_2020.npz', put it into `./data`  
  * Download [PIXIE data](http://pixie.is.tue.mpg.de/): 
      * the pre-trained model 'pixie_model.tar' and place it into `./data`   
      * the utilities file `utilities.zip` and extract its contents into `./data`  
  * (Optional) Follow the instructions for the [Albedo model](https://github.com/TimoBolkart/BFM_to_FLAME) to get 'FLAME_albedo_from_BFM.npz'. Put it into `./data`  
  * (Optional) Clone and prepare [DECA](https://github.com/YadiraF/DECA)
