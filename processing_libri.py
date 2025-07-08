import os
import random
import numpy as np
import soundfile as sf
import json
from datasets import load_dataset
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TARGET_DIR = "../datasets/wav"
METADATA_DIR = "../datasets/metadata"
MIN_DURATION = 5 * 60
MAX_DURATION = 15 * 60
SAMPLE_RATE = 16000  # LibriSpeech sample rate is 16kHz (16000 samples per second), not seconds

def ensure_target_dirs():
    """Ensure the target directories exist."""
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    logger.info(f"Target directories ensured: {TARGET_DIR}, {METADATA_DIR}")

def load_librispeech():
    """Load LibriSpeech dataset from Hugging Face with streaming mode."""
    logger.info("Loading LibriSpeech dataset in streaming mode...")
    # Load the 'clean' subset of LibriSpeech with streaming enabled
    dataset = load_dataset("librispeech_asr", "clean", streaming=True)
    return dataset

def group_by_speaker_and_chapter(dataset):
    """Group audio samples by speaker and chapter from a streaming dataset."""
    logger.info("Grouping samples by speaker and chapter from streaming dataset...")
    groups = defaultdict(list)
    sample_count = 0

    # LibriSpeech 'clean' contains these splits
    splits = ['train.clean.100', 'train.clean.360', 'test.clean', 'dev.clean']

    for split in splits:
        logger.info(f"Processing split: {split}")
        split_dataset = dataset[split]

        for item in split_dataset:
            speaker_id = item["speaker_id"]
            chapter_id = item["chapter_id"]
            key = (speaker_id, chapter_id)
            groups[key].append(item)
            sample_count += 1

            if sample_count % 1000 == 0:
                logger.info(f"Processed {sample_count} samples so far...")

    logger.info(f"Processed a total of {sample_count} samples")
    logger.info(f"Found {len(groups)} unique speaker-chapter combinations")
    return groups

def concatenate_audio(audio_list):
    """Concatenate a list of audio arrays into a single array."""
    return np.concatenate(audio_list)

def create_concatenated_samples(groups, target_duration_range=(MIN_DURATION, MAX_DURATION)):
    """
    Create concatenated audio samples within the target duration range.
    Only concatenates audio from the same speaker reading the same chapter.
    """
    logger.info("Creating concatenated samples...")
    min_duration, max_duration = target_duration_range
    concatenated_samples = []

    for (speaker_id, chapter_id), items in groups.items():
        # Sort items by id to maintain order
        items.sort(key=lambda x: x["id"])

        # Calculate total duration for this speaker-chapter combination
        total_duration = sum(len(item["audio"]["array"]) / SAMPLE_RATE for item in items)

        if total_duration < min_duration:
            logger.debug(f"Speaker {speaker_id}, Chapter {chapter_id}: Total duration {total_duration:.2f}s is less than minimum {min_duration}s, skipping")
            continue

        # Initialize variables for concatenation
        current_samples = []
        current_duration = 0

        for item in items:
            audio_array = item["audio"]["array"]
            item_duration = len(audio_array) / SAMPLE_RATE

            # Add this item to the current batch
            current_samples.append(audio_array)
            current_duration += item_duration

            # If we've reached the minimum duration, check if we should create a sample
            if current_duration >= min_duration:
                # If we're over the maximum duration, create a sample and reset
                if current_duration > max_duration:
                    # Create a sample with the current batch
                    concatenated_audio = concatenate_audio(current_samples)
                    concatenated_samples.append({
                        "audio": concatenated_audio,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "duration": current_duration
                    })
                    logger.info(f"Created sample: Speaker {speaker_id}, Chapter {chapter_id}, Duration {current_duration:.2f}s")

                    # Reset for the next batch
                    current_samples = []
                    current_duration = 0
                # If we're within the range, randomly decide whether to add more or create a sample
                elif random.random() < 0.5:  # 50% chance to create a sample if within range
                    concatenated_audio = concatenate_audio(current_samples)
                    concatenated_samples.append({
                        "audio": concatenated_audio,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "duration": current_duration
                    })
                    logger.info(f"Created sample: Speaker {speaker_id}, Chapter {chapter_id}, Duration {current_duration:.2f}s")

                    # Reset for the next batch
                    current_samples = []
                    current_duration = 0

        # Don't forget any remaining audio if it meets the minimum duration
        if current_duration >= min_duration:
            concatenated_audio = concatenate_audio(current_samples)
            concatenated_samples.append({
                "audio": concatenated_audio,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "duration": current_duration
            })
            logger.info(f"Created sample: Speaker {speaker_id}, Chapter {chapter_id}, Duration {current_duration:.2f}s")

    logger.info(f"Created {len(concatenated_samples)} concatenated samples")
    return concatenated_samples

def translate_to_chinese(text):
    """
    Translate text to Chinese.
    This is a placeholder function. In a real implementation, you would call
    an actual translation API or service.
    """
    logger.info(f"Translating text to Chinese: {text[:50]}...")
    # Placeholder for translation API call
    # In a real implementation, you would use a service like Google Translate,
    # Baidu Translate, or another translation API
    return f"[Chinese translation of: {text}]"

def save_samples(samples):
    """
    Save concatenated audio samples to the target directory with streaming processing.
    Also exports metadata in JSON format. If metadata.json exists, it will be loaded and updated,
    otherwise a new one will be initialized.
    """
    logger.info(f"Saving {len(samples)} samples to {TARGET_DIR} with streaming processing...")

    # Define metadata file path
    metadata_file = os.path.join(METADATA_DIR, "metadata.json")

    # Initialize or load existing metadata
    metadata_entries = []
    if os.path.exists(metadata_file):
        logger.info(f"Loading existing metadata from {metadata_file}")
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f_meta:
                metadata_entries = json.load(f_meta)
        except json.JSONDecodeError:
            logger.warning(f"Error decoding {metadata_file}, initializing new metadata")
            metadata_entries = []
    else:
        logger.info(f"Initializing new metadata file at {metadata_file}")

    # Process and save samples
    for i, sample in enumerate(samples):
        # Create filename for audio using dataset name + six-digit number format
        file_number = len(metadata_entries) + i + 1
        filename = f"librispeech_{file_number:06d}.wav"
        filepath = os.path.join(TARGET_DIR, filename)

        # Save the audio file with streaming processing
        # Note: soundfile's write function already handles streaming for large files
        sf.write(filepath, sample["audio"], SAMPLE_RATE)

        # Create metadata entry
        metadata = {
            "audio_file": filename,
            "speaker_id": sample["speaker_id"],
            "chapter_id": sample["chapter_id"],
            "duration": sample["duration"],
            # Add more metadata fields as needed
        }

        # Add to metadata entries
        metadata_entries.append(metadata)

        logger.info(f"Processed sample {i+1}/{len(samples)}: {filepath} ({sample['duration']:.2f}s)")

    # Save updated metadata
    with open(metadata_file, 'w', encoding='utf-8') as f_meta:
        json.dump(metadata_entries, f_meta, indent=2)

def main():
    """Main function to process LibriSpeech dataset with streaming."""
    # Ensure target directories exist
    ensure_target_dirs()

    # Load dataset in streaming mode
    dataset = load_librispeech()

    # Group by speaker and chapter from streaming dataset
    groups = group_by_speaker_and_chapter(dataset)

    # Create concatenated samples
    samples = create_concatenated_samples(groups)

    # Save samples with streaming processing and metadata export
    save_samples(samples)

    logger.info("Streaming processing complete!")

if __name__ == "__main__":
    main()
