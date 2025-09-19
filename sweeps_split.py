import os
import json
from nuscenes.nuscenes import NuScenes
from find_token_and_segment import save_segment_metadata

def split_all_sweeps(nusc, image_root_dir, split='trainval'):
    import os, json
    
    split_dir = os.path.join(image_root_dir, "split")
    #split_dir = os.path.join(os.getcwd(), "sweeps_split")  # for testing
    os.makedirs(split_dir, exist_ok=True)

    # load sample_data.json，create the map of token → entry
    with open(os.path.join(nusc.dataroot, 'v1.0-'+split+'/sample_data.json')) as f:
        sample_data_entries = json.load(f)
    sample_data_map = {entry['token']: entry for entry in sample_data_entries}

    scene_count = 0

    for scene in nusc.scene:
        # scene_id = f"{scene_count:04d}"
        scene_name = scene['name']            
        scene_id = scene_name.split('-')[1]   
        scene_output_dir = os.path.join(split_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)

        print(f"\nprocessing the {scene_id} th scene：{scene['name']}")

        first_sample_token = scene['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        cam_front_token = first_sample['data']['CAM_FRONT']

        ordered_tokens = []
        filenames = []

        current_token = cam_front_token
        while current_token:
            entry = sample_data_map.get(current_token)
            if entry is None:
                print(f"can't find sample_data token：{current_token}")
                break

            if not entry['is_key_frame']:
                ordered_tokens.append(entry['token'])
                filenames.append(entry['filename'])

            current_token = entry['next']

        json_filename = f"{scene_id}.json"

        save_segment_metadata(
            nusc=nusc,
            folder_path=scene_output_dir,
            ordered_tokens=ordered_tokens,
            output_file_name=json_filename,
            filenames=filenames,
            sample_data=sample_data_entries,
            split=split
        )

if __name__ == "__main__":
    split = 'trainval'
    DATASET_DIR = "/data/NuScene/v1.0-"+split+"_meta"
    IMAGE_ROOT = "/data/NuScene/v1.0-"+split+"/sweeps/CAM_FRONT"
    nusc = NuScenes(version='v1.0-'+split, dataroot=DATASET_DIR, verbose=True)

    split_all_sweeps(nusc, IMAGE_ROOT, split)
