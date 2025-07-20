import os
import json
import shutil
import argparse
import wave

class DatasetValidator:
    """
    A class to check the validity of a dataset or roll it back to a clean state.
    almost manipulates/provide the protential fix of the manifast and wav files.
    The dataset is expected to have the following structure:
    - folder/
        - metadata.json
        - all_audios.jsonl
        - wavs/
            - 000000.wav
            - 000001.wav
            - ...
    """

    def __init__(self, folder_path):
        """
        Initializes the validator with the path to the dataset folder.

        Args:
            folder_path (str): The path to the dataset folder.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The specified directory does not exist: {folder_path}")

        self.folder_path = folder_path
        self.metadata_path = os.path.join(folder_path, 'metadata.json')
        self.manifest_path = os.path.join(folder_path, 'all_audios.jsonl')
        self.wavs_path = os.path.join(folder_path, 'wavs')

        # Define the expected schema for each record in the manifest
        self.schema = {
            'id': str,
            'source_ds': str,
            'duration_sec': float,
            'audio_auto': bool,
            'text_auto': bool,
            'num_speakers': int,
            'num_switches': int,
            'lang': str,
            'slice': list,
            'transcribe': str,
            'components': list
        }

    def _validate_record_schema(self, record, line_num):
        """
        Validates the schema of a single record from the manifest file.
        """
        # 1. Check for missing or extra keys
        record_keys = set(record.keys())
        expected_keys = set(self.schema.keys())
        if record_keys != expected_keys:
            missing = expected_keys - record_keys
            extra = record_keys - expected_keys
            error_msg = f"Schema mismatch at line {line_num}."
            if missing:
                error_msg += f" Missing keys: {missing}."
            if extra:
                error_msg += f" Extra keys: {extra}."
            raise AssertionError(error_msg)

        # 2. Check data types and specific formats for each key
        for key, expected_type in self.schema.items():
            if not isinstance(record[key], expected_type):
                raise AssertionError(f"Invalid type for key '{key}' at line {line_num}. "
                                     f"Expected {expected_type}, got {type(record[key])}.")

        # 3. Perform more detailed format validation
        # 'slice': must be a list of lists, each with two numbers
        if not all(isinstance(s, list) and len(s) == 2 and all(isinstance(x, (float, float)) for x in s) for s in record['slice']):
            raise AssertionError(f"Invalid format for 'slice' at line {line_num}. "
                                 f"Expected a list of [number, number] pairs.")

        # 'components': must be a list of strings
        if not all(isinstance(c, str) for c in record['components']):
            raise AssertionError(f"Invalid format for 'components' at line {line_num}. "
                                 f"Expected a list of strings.")


    def check_validity_manifact(self):
        """
        Performs a comprehensive check of the dataset.

        - Checks if file counts (wavs, manifest lines) match the metadata.
        - Validates the schema of every record in the manifest file.
        - Ensures all audio IDs are unique, sequential, and within the valid range.
        - Verifies that every expected .wav file exists.
        """

        # 1. Check file and folder existence
        for path in [self.metadata_path, self.manifest_path, self.wavs_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file or directory not found: {path}")

        # 2. Check counts from metadata and file system
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            amount_ideal = json.load(f)['amount']
        amount_wavs = len(os.listdir(self.wavs_path))

        # 3. Check each line in the manifest file
        valid_json_count = 0
        seen_ids = set()
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    valid_json_count += 1

                    # Validate the schema of the record
                    self._validate_record_schema(record, line_num)

                    # Check if the ID is an integer in the valid range
                    json_id = int(record['id'])
                    if not (0 <= json_id < amount_ideal):
                        raise AssertionError(f"JSON at line {line_num} has ID {json_id} which is outside the valid range [0, {amount_ideal}).")

                    # Check for duplicate IDs
                    if json_id in seen_ids:
                        raise AssertionError(f"Duplicate ID {json_id} found at line {line_num}.")
                    seen_ids.add(json_id)

                except json.JSONDecodeError:
                    raise AssertionError(f"Invalid JSON at line {line_num} in {self.manifest_path}.")
                except (KeyError, ValueError) as e:
                    raise AssertionError(f"Error processing record at line {line_num}: {e}")


        amount_manifests = valid_json_count
        assert amount_ideal == amount_wavs == amount_manifests, \
            f'Count mismatch! Metadata: {amount_ideal}, WAVs: {amount_wavs}, Manifest lines: {amount_manifests}.'

        # Check if a corresponding .wav file exists for each ID


        print('✅ All good! Dataset is valid.')

    def get_ill_waveforms(self):
        """
        fix the wav files in the dataset according to the manifact.
        Return:
            [str]: List of id to be fixed.
        """
        ill_wavs = []
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            amount_ideal = json.load(f)['amount']
        for i in range(amount_ideal):
            wav_file = os.path.join(self.wavs_path, f'{i:06d}.wav')
            assert os.path.exists(wav_file), f'{wav_file} not found.'
            try:
                with wave.open(wav_file, 'rb') as f:
                    f.readframes(1)
            except wave.Error as e:
                print(f"Error reading {wav_file}: {e}")
                ill_wavs.append(f'{i:06d}')

        return ill_wavs


    def rollback(self):
        """
        Rolls back the dataset to the state defined by 'metadata.json'.

        - Deletes any .wav files with IDs greater than or equal to the ideal amount.
        - Truncates the manifest file to only include records with IDs less than the ideal amount.
        """
        print("⏪ Starting dataset rollback...")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            amount_ideal = json.load(f)['amount']

        # 1. Delete extra .wav files
        print(f"Ideal amount is {amount_ideal}. Checking for extra .wav files...")
        deleted_wav_count = 0
        for filename in os.listdir(self.wavs_path):
            if filename.endswith('.wav'):
                try:
                    file_id = int(filename.split('.')[0])
                    if file_id >= amount_ideal:
                        os.remove(os.path.join(self.wavs_path, filename))
                        deleted_wav_count += 1
                except (ValueError, IndexError):
                    print(f"  - Skipping non-standard wav file: {filename}")
        print(f"  - Deleted {deleted_wav_count} extra .wav files.")

        # 2. Filter the manifest file
        print(f"Filtering manifest file: {self.manifest_path}")
        temp_manifest_path = self.manifest_path + '.tmp'
        kept_lines = 0
        with open(self.manifest_path, 'r', encoding='utf-8') as infile, \
             open(temp_manifest_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    item = json.loads(line)
                    if int(item.get('id', -1)) < amount_ideal:
                        outfile.write(line)
                        kept_lines += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    print(f"  - Skipping malformed line in manifest: {line.strip()}")

        shutil.move(temp_manifest_path, self.manifest_path)
        print("✅ Rollback complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform checks or rollback operations on a dataset.")
    parser.add_argument(
        'action',
        type=str,
        choices=['check', 'rollback'],
        help="The action to perform: 'check' for enhanced validation or 'rollback'."
    )
    parser.add_argument(
        'path',
        type=str,
        help="The path to the target dataset directory."
    )

    args = parser.parse_args()

    try:
        validator = DatasetValidator(args.path)
        if args.action == 'check':
            validator.check_validity_manifact()
        elif args.action == 'rollback':
            validator.rollback()
    except (FileNotFoundError, AssertionError) as e:
        print(f"\n❌ ERROR: {e}")