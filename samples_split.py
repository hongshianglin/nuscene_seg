import os
import json
from nuscenes.nuscenes import NuScenes
from find_token_and_segment import save_segment_metadata

def split_all_samples(nusc, image_root_dir):
    split_dir = os.path.join(image_root_dir, "split")
    #split_dir = os.path.join(os.getcwd(), "samples_split") # for testing
    os.makedirs(split_dir, exist_ok=True)

    scene_count = 0

    with open(os.path.join(nusc.dataroot, 'v1.0-trainval/sample_data.json')) as f:
        sample_data = json.load(f)

    for scene in nusc.scene:
        # scene_count += 1

        """
        # adjust the start point of scene
        if scene_count < 615:
            continue
        """

        # scene_id = f"{scene_count:04d}"
        scene_name = scene['name']
        scene_id = scene_name.split('-')[1]
        scene_output_dir = os.path.join(split_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)

        print(f"\nprocessing the {scene_id} th scene：{scene['name']}")

        current_sample_token = scene['first_sample_token']
        ordered_tokens = []
        filenames = []

        while current_sample_token:
            sample = nusc.get('sample', current_sample_token)
            cam_front_token = sample['data']['CAM_FRONT']
            ordered_tokens.append(cam_front_token)

            sd = nusc.get('sample_data', cam_front_token)
            filenames.append(sd['filename'])

            current_sample_token = sample['next']

        json_filename = f"{scene_id}.json"
        save_segment_metadata(
            nusc=nusc,
            folder_path=scene_output_dir,
            ordered_tokens=ordered_tokens,
            output_file_name=json_filename,
            filenames=filenames,
            sample_data=sample_data
        )

if __name__ == "__main__":
    DATASET_DIR = "/data/NuScene/v1.0-trainval_meta"
    IMAGE_ROOT = "/data/NuScene/v1.0-trainval/samples/CAM_FRONT"
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_DIR, verbose=True)
    
    split_all_samples(nusc, IMAGE_ROOT)
