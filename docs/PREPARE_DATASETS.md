# Dataset Preparation Instructions

### NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Generate the data infos by running the following command (it may take several hours): 
```python
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval --load_interval 4
```

**Note:** NuScenes collect LiDAR data with 20Hz and denote keyframe with 2Hz. Thus, a load interval of 4 on keyframe create a 2.5% subset of NuScenes dataset.

### Simulation Dataset