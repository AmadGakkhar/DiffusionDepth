#!/usr/bin/env python3
"""
Quick semantic toggle test
Add this to your training script to easily compare with/without semantics
"""

# Add this flag at the top of your training script
USE_SEMANTIC = True  # Change to False to disable semantic processing

# In your model's forward method, modify the extract_depth call:
def forward(self, sample):
    # ... existing code ...
    
    # Toggle semantic usage for quick testing
    semantic_input = sample.get('semantic', None) if USE_SEMANTIC else None
    
    output_dict = self.extract_depth(
        img_inputs, depth_map, depth_mask, gt_depth_map, 
        img_metas=None, return_loss=True, weight_map=None, 
        instance_masks=None, semantic=semantic_input, 
        sparse_depth=sparse_depth
    )
    return output_dict

print(f"Semantic processing: {'ENABLED' if USE_SEMANTIC else 'DISABLED'}") 