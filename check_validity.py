import json
import os
import shutil
import argparse

def check_validity(folder):
    # check if the the number of wavs files is equal to the number of txt files also the metadata
    amount_ideal = json.load(open(folder + '/metadata.json'))['amount']
    amount_wavs = len(os.listdir(folder + '/wavs'))

    # Check if each line in all_audios.jsonl is a valid JSON and IDs fill the range [0, amount_ideal)
    valid_json_count = 0
    seen_ids = set()
    with open(os.path.join(folder, 'all_audios.jsonl'), 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                json_obj = json.loads(line)
                valid_json_count += 1

                # Check if the ID is in the valid range
                json_id = int(json_obj['id'])
                if json_id < 0 or json_id >= amount_ideal:
                    raise AssertionError(f"JSON at line {line_num} has ID {json_id} which is outside the valid range [0, {amount_ideal})")

                # Check for duplicate IDs
                if json_id in seen_ids:
                    raise AssertionError(f"Duplicate ID {json_id} found at line {line_num}")

                seen_ids.add(json_id)
            except json.JSONDecodeError:
                raise AssertionError(f"Invalid JSON at line {line_num} in all_audios.jsonl")

    # Check if all IDs in the range [0, amount_ideal) are present
    missing_ids = set(range(amount_ideal)) - seen_ids
    if missing_ids:
        raise AssertionError(f"Missing IDs in the range [0, {amount_ideal}): {missing_ids}")

    amount_manifacts = valid_json_count
    assert amount_ideal == amount_wavs == amount_manifacts, f'length wrong {amount_ideal} != {amount_wavs} != {amount_manifacts}'

    #检查每一条seq number对应的wav都存在
    for i in range(amount_ideal):
        assert os.path.exists(os.path.join(folder, 'wavs', f'{i:06d}.wav')), f'{i:06d}.wav not found'

    print('All good!')

def roll_back(folder):
    amount_ideal = json.load(open(folder + '/metadata.json'))['amount']
    amount_wavs = len(os.listdir(folder + '/wavs'))
    with open(os.path.join(folder, 'all_audios.jsonl'), 'r', encoding='utf-8') as f:
        amount_manifacts = sum(1 for line in f)
    # delete all the wavs that larger than this amount
    for i in range(amount_ideal, amount_wavs):
        os.remove(os.path.join(folder, 'wavs', f'{i:06d}.wav'))

    # delete all the manifests that larger than this amount
    manifest_path = os.path.join(folder, 'all_audios.jsonl')
    temp_manifest_path = os.path.join(folder, 'tmp_all_audios.jsonl')
    print(f"Filtering manifest file: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as infile, \
            open(temp_manifest_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Keep the line only if the sequence number is LESS THAN the ideal amount
            item = json.loads(line)
            if int(item.get('id')) < amount_ideal:
                outfile.write(line)

    # Replace the original file with the filtered temporary file
    shutil.move(temp_manifest_path, manifest_path)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="对数据集执行检查或回滚操作。")
    parser.add_argument(
        'action',
        type=str,
        choices=['check', 'rollback'],
    )
    args = parser.parse_args()
    if args.action == 'check':
        check_validity('../datasets/LongSpeech_p2')
    elif args.action == 'rollback':
        roll_back('../datasets/LongSpeech_p2')
