import json
import os
import shutil

def check_validity(folder):
    # check if the the number of wavs files is equal to the number of txt files also the metadata
    amount_ideal = json.load(open(folder + '/metadata.json'))['amount']
    amount_wavs = len(os.listdir(folder + '/wavs'))
    with open(os.path.join(folder, 'all_audios.jsonl'), 'r', encoding='utf-8') as f:
        amount_manifacts = sum(1 for line in f)
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
    check_validity('../datasets/LongSpeech')
    #roll_back('../datasets/LongSpeech')