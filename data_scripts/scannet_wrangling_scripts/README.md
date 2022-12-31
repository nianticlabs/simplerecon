# Downloading and Extracting ScanNetv2


Developed and tested with python 3.9.

The included license at LICENSE applies only to `reader.py` and `SensorData.py`.


These scripts should help you export ScanNetv2 to the following format:

    SCANNET_ROOT
        scans_test (test scans)
            scene0707
                scene0707_00_vh_clean_2.ply (gt mesh)
                sensor_data
                    frame-000261.pose.txt
                    frame-000261.color.jpg 
                    frame-000261.color.512.png (optional, image at 512x384)
                    frame-000261.color.640.png (optional, image at 640x480)
                    frame-000261.depth.png (full res depth, stored scale *1000)
                    frame-000261.depth.256.png (optional, depth at 256x192 also
                                                scaled)
                scene0707.txt (scan metadata and image sizes)
                intrinsic
                    intrinsic_depth.txt
                    intrinsic_color.txt

            ...
        scans (val and train scans)
            scene0000_00
                (see above)
            scene0000_01
            ....

Make sure all the packages in `env.yml` are installed in your environment.

## Downloading ScanNetv2

The `download_scannet.py` script is from https://kaldir.vc.in.tum.de/scannet/download-scannet.py

Please make sure you fill in this form before downloading the data:
https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf

Download the dataset by running:
```
python download_scannet.py -o SCANNET_ROOT
```

For one scan debug use:
```
python download_scannet.py -o SCANNET_ROOT --id scene0707_00
```

This will download a `.sens` file, `.txt` file, the high resolution mesh `ply`, and a lower resolution mesh `ply`. 

`.txt` will include meta information for the scan. See the next section for extracting the `.sens` file.

## Extracting data from .sens files

Please use the intrinsics directly from the downloaded `.txt` file from the dataset.

This is a modified version of the SensReader python script at 
https://github.com/ScanNet/ScanNet/tree/master/SensReader/python


`reader.py` will extract depth, jpg, and intrinics files from ScanNetv2's downloaded `.sens` files. It will dump the `jpg` data directly to disk without uncompressing/compressing.

To extract all scans for test:
```
python reader.py --scans_folder SCANNET_ROOT/scans_test \
                 --output_path  OUTPUT_PATH/scans_test \
                 --scan_list_file splits/scannetv2_test.txt \
                 --num_workers 12 \
                 --export_poses \
                 --export_depth_images \
                 --export_color_images \
                 --export_intrinsics;
```

For train and val
```
python reader.py --scans_folder SCANNET_ROOT/scans \
                 --output_path  OUTPUT_PATH/scans \
                 --scan_list_file splits/scannetv2_train.txt \
                 --num_workers 12 \
                 --export_poses \
                 --export_depth_images \
                 --export_color_images \
                 --export_intrinsics;

python reader.py --scans_folder SCANNET_ROOT/scans \
                 --output_path  OUTPUT_PATH/scans \
                 --scan_list_file splits/scannetv2_val.txt \
                 --num_workers 12 \
                 --export_poses \
                 --export_depth_images \
                 --export_color_images \
                 --export_intrinsics;
```

`OUTPUT_PATH` can be the same directory as the ScanNet root directory `SCANNET_ROOT`.

For one scan use `--single_debug_scan_id`.

For caching resized pngs for depth and color files, run:

```
python reader.py --scans_folder SCANNET_ROOT/scans \
                 --output_path OUTPUT_PATH/scans \
                 --scan_list_file splits/scannetv2_train.txt \
                 --num_workers 12 \
                 --export_depth_images \
                 --export_color_images \
                 --rgb_resize 512 384 \
                 --depth_resize 256 192;
```

and for images at `640x480`:

```
python reader.py --scans_folder SCANNET_ROOT/scans \
                 --output_path OUTPUT_PATH/scans \
                 --scan_list_file splits/scannetv2_train.txt \
                 --num_workers 12 \
                 --export_color_images \
                 --rgb_resize 640 480 \
```
