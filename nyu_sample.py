import json
import random

# Set random seed for reproducibility
random.seed(42)

# Load the full NYU dataset split file
with open('data_json/nyu.json', 'r') as f:
    nyu_data = json.load(f)

# Sample 200 entries from each split while preserving the split structure
sampled_data = {}
for split in ['train', 'test', 'val']:
    if split in nyu_data:
        # Get all entries for this split
        split_entries = nyu_data[split]
        
        # Sample 200 entries or all if less than 200 available
        n_samples = min(200, len(split_entries))
        sampled_entries = random.sample(split_entries, n_samples)
        
        # Store sampled entries
        sampled_data[split] = sampled_entries

# Save sampled dataset to new JSON file
output_file = 'data_json/nyu_sampled.json'
with open(output_file, 'w') as f:
    json.dump(sampled_data, f, indent=4)

print(f'Sampled dataset saved to {output_file}')
for split in sampled_data:
    print(f'Number of {split} samples: {len(sampled_data[split])}')
