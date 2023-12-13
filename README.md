# ShapeMoiré
Pytorch implementation of [ShapeMoiré: Channel-Wise Shape-Guided Network for Image Demoiréing].

## Usage


### Environment Installation 

1. Requirements
   - Linux or macOS (Windows is not currently officially supported)
   - Python 3.8
   - PyTorch 1.9.0
   - CUDA 11.1 
   - GCC 7.3.0

2. Install dependencies.

    Here is a full script for setting up ShapeMoiré with conda.

    ```shell
    # build conda environment
    conda create -n shapemoiré python=3.8 -y
    conda activate shapemoiré
    
    # install latest PyTorch prebuilt with the default prebuilt CUDA version 
    conda install pytorch torchvision -c pytorch

    # install other dependencies
    conda install lpips==0.1.4 numpy==1.25.2 opencv_python==4.8.0.76 Pillow==10.0.1 PyYAML==6.0.1 skimage==0.0 tensorboardX==2.6.2.2 thop==0.1.1.post2209072238 tqdm==4.66.1

    ```


### Data Preparation

1. Download Dataset 

    You can download four open datasets: FHDMi, TIP2018, UHDM and LCDMoiré from the Internet. 
    
    Link dataset path under `$ShapeMoiré/data`.

2. Data Structure

    Finally, the total data structure is shown like this:
    ```
    ShapeMoiré/
    |---configs/
    |---data/
    |   |---FHDMi/
    |   |   |---train/
    |   |   |---test/
    |   |---TIP/
    |   |   |---train/
    |   |   |---test/
    |   |---UHDM/
    |   |   |---train/
    |   |   |---test/
    |   |---AIM/ (for LCDMoiré)
    |   |   |---train/
    |   |   |---test/
    |---dataset/
    |---model/
    |---utils/
    |---train.py
    |---test.py
    ```

### Modeling 
1. Train

    For training process, we use config file in `$ShapeMoiré/configs` to define model, dataset and hyber parameters.

    Run the following command to start a training process. You should **specify the model and dataset before training**. 

    ```bash
    python train_{MODEL_NAME}.py --config configs/{MODEL_NAME}/{DATASET_NAME}.yaml
    ```

    Note: 
    * **The default config file is defined to train ShapeMoiré**. For training baseline model, modify config with 'USE_BASELINE: True'.
    * For ESDNet, in order to train ESDNet-L, modify config with 'SAM_NUM:2'.


2. Test

    Run the following command to start a testing process. 

    Except for choosing model and dataset, You need to **specify the checkpoint using the parameter 'LOAD_PATH'** within config. 

    ```bash 
    python test_{MODEL_NAME}.py --config configs/{MODEL_NAME}/{DATASET_NAME}.yaml
    ```


## Main Results




<table  style="text-align:center">
    <tr >
        <th rowspan="2" style="text-align:center">Architecture</th><th rowspan="2" style="text-align:center">Method</th><th colspan="4" style="text-align:center">PSNR</th><th rowspan="2" style="text-align:center">Params. (M)</th>
    </tr>
    <tr>
        <th>UHDM</th><th>FHDMi</th><th>TIP2018</th><th>LCDMoiré</th>
    </tr>
    <tr>
        <td rowspan="3">ESDNet</td><td>Baseline</td><td>22.253</td><td>24.393</td><td>29.791</td><td>45.286</td><td>5.394</td>
    </tr>
    <tr>
        <td>ShapeMoiré</td><td>22.597</td><td>24.629</td><td>29.862</td><td>45.537</td><td>5.394</td>
    </tr>
    <tr>
        <td>+</td><td>0.344</td><td>0.236</td><td>0.071</td><td>0.251</td><td>0</td>
    </tr>
    <tr>
        <td rowspan="3">ESDNet-L</td><td>Baseline</td><td>22.554</td><td>24.808</td><td>30.096</td><td>45.544</td><td>10.623</td>
    </tr>
    <tr>
        <td>ShapeMoiré</td><td>22.948</td><td>25.064</td><td>30.161</td><td>46.558</td><td>10.623</td>
    </tr>
    <tr>
        <td>+</td><td>0.394</td><td>0.256</td><td>0.065</td><td>1.014</td><td>0</td>
    </tr>
    <tr>
        <td rowspan="3">WDNet</td><td>Baseline</td><td>19.182</td><td>21.161</td><td>27.812</td><td>37.324</td><td>3.360</td>
    </tr>
    <tr>
        <td>ShapeMoiré</td><td>19.882</td><td>22.182</td><td>28.312</td><td>38.408</td><td>3.360</td>
    </tr>
    <tr>
        <td>+</td><td>0.500</td><td>1.021</td><td>0.500</td><td>1.084</td><td>0</td>
    </tr>
    <tr>
        <td rowspan="3">DMCNN</td><td>Baseline</td><td>17.812</td><td>19.313</td><td>24.519</td><td>29.321</td><td>1.426</td>
    </tr>
    <tr>
        <td>ShapeMoiré</td><td>18.036</td><td>19.615</td><td>25.381</td><td>29.649</td><td>1.426</td>
    </tr>
    <tr>
        <td>+</td><td>0.223</td><td>0.302</td><td>0.862</td><td>0.329</td><td>0</td>
    </tr>
    
</table>







