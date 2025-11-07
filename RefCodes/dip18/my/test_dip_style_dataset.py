"""
Quick test script to verify DIP-style dataset loading works correctly.
This script loads a small sample of data and prints statistics.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from my.dataset_dip_style import DIPStyleDataset, collate_fn_dip_style


def test_dataset_loading():
    """Test basic dataset loading."""
    print("="*60)
    print("Testing DIP-Style Dataset Loading")
    print("="*60)
    print()
    
    # Configuration
    datasets = ['processed_seg_data_BEHAVE']  # Use a single dataset for testing
    data_root = '../../process'
    subset = 'train'
    
    print(f"Loading dataset: {datasets}")
    print(f"Data root: {data_root}")
    print(f"Subset: {subset}")
    print()
    
    try:
        # Create dataset
        dataset = DIPStyleDataset(
            dataset_names=datasets,
            data_root=data_root,
            subset=subset,
            seq_len=120,
            random_sample=True,
            use_full_sequence=False,
            fps_override=30.0,
            trim_frames=6,
            imu_noise_std=0.1,
            normalize=True,
            data_stats=None,
        )
        
        print("✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Human IMU dimension: {dataset.human_input_dim}")
        print(f"  Object IMU dimension: {dataset.object_input_dim}")
        print(f"  Human pose dimension: {dataset.human_pose_dim}")
        print(f"  Object velocity dimension: {dataset.object_velocity_dim}")
        print()
        
        # Test single sample
        print("Testing single sample retrieval...")
        sample = dataset[0]
        print("✓ Sample retrieved successfully")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Human IMU shape: {sample['human_imu'].shape}")
        print(f"  Object IMU shape: {sample['object_imu'].shape}")
        print(f"  Human pose shape: {sample['human_pose'].shape}")
        print(f"  Object velocity shape: {sample['object_velocity'].shape}")
        print(f"  Object position shape: {sample['object_position'].shape}")
        print(f"  Sequence length: {sample['seq_len']}")
        print()
        
        # Test data statistics
        print("Checking data statistics...")
        stats = dataset.get_statistics()
        print("✓ Statistics computed")
        print(f"  Available statistics: {list(stats.keys())}")
        for key, stat_dict in stats.items():
            print(f"  {key}:")
            print(f"    Mean shape: {stat_dict['mean_channel'].shape}")
            print(f"    Std shape: {stat_dict['std_channel'].shape}")
            print(f"    Mean range: [{stat_dict['mean_channel'].min():.4f}, {stat_dict['mean_channel'].max():.4f}]")
            print(f"    Std range: [{stat_dict['std_channel'].min():.4f}, {stat_dict['std_channel'].max():.4f}]")
        print()
        
        # Test DataLoader
        print("Testing DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn_dip_style,
            drop_last=False,
        )
        
        batch = next(iter(loader))
        print("✓ DataLoader works")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Batch size: {batch['human_imu'].shape[0]}")
        print(f"  Max sequence length in batch: {batch['human_imu'].shape[1]}")
        print(f"  Human IMU batch shape: {batch['human_imu'].shape}")
        print(f"  Object IMU batch shape: {batch['object_imu'].shape}")
        print(f"  Sequence lengths: {batch['seq_len']}")
        print()
        
        # Test sample generator
        print("Testing sample generator...")
        gen = dataset.sample_generator()
        sample_from_gen = next(gen)
        print("✓ Sample generator works")
        print(f"  Generated sample keys: {list(sample_from_gen.keys())}")
        print()
        
        # Test normalization undo
        print("Testing normalization undo...")
        pose_normalized = sample['human_pose'].numpy()
        pose_unnormalized = dataset.undo_normalization_pose(pose_normalized)
        print("✓ Normalization undo works")
        print(f"  Normalized pose range: [{pose_normalized.min():.4f}, {pose_normalized.max():.4f}]")
        print(f"  Unnormalized pose range: [{pose_unnormalized.min():.4f}, {pose_unnormalized.max():.4f}]")
        print()
        
        print("="*60)
        print("✓ All tests passed!")
        print("="*60)
        return True
        
    except FileNotFoundError as e:
        print(f"✗ Error: Data directory not found")
        print(f"  {e}")
        print(f"  Please check that the data path is correct:")
        print(f"    {Path(data_root).absolute() / datasets[0] / subset}")
        return False
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)

