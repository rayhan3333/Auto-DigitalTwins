## Extract Frames
```bash
python data_tools/extract_frames.py \
    --video_dir ./data/videos \
    --video_name ./microwave \
    --data_dir ./data/videoartgs/realscan \
    --interval 2 \
    --resize 2
```
where ``video_dir`` is the path to the video directory, ``video_name`` is the name of the video, ``data_dir`` is the path to the scene directory, ``interval`` is the interval between frames (default: 2), and ``resize`` is the resize factor (default: 2). We assume that the object is static in the first N_cano(default: 100) frames. All extracted frames will be saved in the ``data_dir/video_name/images`` folder.
## Predict Depth & Camera Pose
In our experiments, we use [SpatialTrackerV2](https://spatialtracker.github.io/) to predict the depth and camera pose. We recommand to use [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3) for more robust results. You can run the following script to predict the depth and camera pose:
```bash
conda activate da3
python data_tools/infer_da3.py \
    --data_dir ./data/videoartgs/realscan \
    --video_name microwave_ego \
    --reprocess
```
or
```bash
conda activate st2
python third_party/SpatialTrackerV2/infer_st2.py \
    --data_dir ./data/videoartgs/realscan \
    --video_name microwave_ego \
    --reprocess
```
All predicted results will be saved in ``data_dir/video_name/da3_result.npz`` or ``data_dir/video_name/st2_result.npz`` respectively.

Note that you need to build the ``da3`` or ``st2`` conda environment and download the checkpoints before running the script. You can build the environment following the instructions in the [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3) and [SpatialTrackerV2](https://spatialtracker.github.io/) repositories.

You can visualize the predicted depth and camera pose using the following script:
```bash
conda activate st2
python third_party/SpatialTrackerV2/tapip3d_viz.py data_dir/video_name/st2_result.npz
python third_party/SpatialTrackerV2/tapip3d_viz.py data_dir/video_name/da3_result.npz
```

## Predict Foreground Mask
We use [SAM2](https://github.com/facebookresearch/sam2) to predict the foreground object mask. We also privide a [tool](https://github.com/YuLiu-LY/sam2-annotation-tools) for GUI operation. All segmented masks should be saved in ``data_dir/video_name/masks`` folder.
We recommend to segment the humans or hands for better results. Segment the foreground object with ID 0 and then segment the humans or hands with ID 1. See the demo video [here](https://github.com/YuLiu-LY/sam2-annotation-tools/blob/main/use_case.mp4).

Then run following scripts to process the data:
```bash
conda activate videoartgs
python data_tools/process_vggt.py \
    --data_dir ./data/videoartgs/realscan \
    --video_name microwave_ego \
    --model da3 \
    --reprocess \
    --visualize 
```
Processed data will be saved in ``data_dir/video_name/data.npz``.

## Predict Joint Count and Types by VLM
We use GPT-4o to predict joint count and types as following format:
```
[
    {
        "id": 0,
        "name": "cabinet_base",
        "joint": "fixed",
        "parent": -1
    },
    {
        "id": 1,
        "name": "drawer",
        "joint": "slider",
        "parent": 0
    },
    {
        "id": 2,
        "name": "door",
        "joint": "hinge",
        "parent": 0
    }
]
```
and save it as ``data_dir/video_name/joint_infos_vlm.json``. 
We provide a script to predict the joint count and types by VLM:
```bash
python data_tools/vlm_process.py \
    --api_key <your_api_key> \
    --base_url <your_base_url>
    --data_dir ./data \
    --dataset videoartgs \
    --subset realscan \
    --video_name microwave_ego \
```
Sometimes, the predicted joint count and types are not correct, you can manually edit the ``joint_infos_vlm.json`` file to correct the errors, making sure the number and types of joints are correct. You can also directly write a json file with the above format.

## Predict 3D Tracks and Analyze Motion
We use [TAPIP3D](https://github.com/zbw001/TAPIP3D) to predict the 3D tracks. Download the checkpoints from [here](https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth) and put it in the ``third_party/TAPIP3D/checkpoints`` folder.
You can run the following script to predict the 3D tracks and analyze the motion:
```bash
python data_tools/extract_tapip3d_track.py \
    --data_dir ./data/videoartgs/realscan \
    --video_name microwave_ego \
    --reprocess 
```
Filtered tracks will be saved in ``data_dir/video_name/filtered.npz`` and joint infos from analyzation will be saved in ``data_dir/video_name/joint_info.json``.
We also save the raw tracks in ``data_dir/video_name/video_name.n<nq>.npz``, where ``nq`` is the number of query frames. 
You can also visualize the raw/filtered tracks using the following script:
```bash
python third_party/TAPIP3D/visualize.py video_name.n<nq>.npz
python third_party/TAPIP3D/visualize.py filtered_vis.npz
```