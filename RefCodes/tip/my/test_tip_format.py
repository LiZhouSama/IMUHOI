"""
Quick test script to verify TIP-format dataset and training pipeline.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from my.dataset_omomo_tip_v2 import OMOMODatasetTIPFormat
from simple_transformer_with_state import TF_RNN_Past_State
from learning_utils import loss_q_only_2axis, loss_jerk


def test_dataset():
    """Test dataset loading and format."""
    print("="*80)
    print("Testing Dataset...")
    print("="*80)
    
    # Test with a small data directory
    data_dirs = ['../../process/processed_data_OMOMO/train']
    
    try:
        dataset = OMOMODatasetTIPFormat(
            data_dirs=data_dirs,
            seq_len=60,
            frame_rate=30.0,
            use_object_imu=True,
            with_acc_sum=False,
            random_sample=True,
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   - Number of sequences: {len(dataset.IMU)}")
        print(f"   - Number of samples: {len(dataset)}")
        print(f"   - Input IMU dim: {dataset.input_imu_dim}")
        print(f"   - State dim: {dataset.state_dim}")
        
        if len(dataset) > 0:
            x_imu, x_s, y_s_n = dataset[0]
            
            print(f"\nSample shapes:")
            print(f"   - x_imu: {x_imu.shape} (expected: [60, 63])")
            print(f"   - x_s: {x_s.shape} (expected: [60, 129])")
            print(f"   - y_s_n: {y_s_n.shape} (expected: [60, 129])")
            
            # Verify shapes
            assert x_imu.shape[0] == 60, f"IMU time dim wrong: {x_imu.shape[0]}"
            assert x_s.shape[0] == 60, f"State time dim wrong: {x_s.shape[0]}"
            assert y_s_n.shape[0] == 60, f"Target time dim wrong: {y_s_n.shape[0]}"
            assert x_imu.shape[1] == 63, f"IMU feature dim wrong: {x_imu.shape[1]} (expected 63 = 7 IMUs × 9)"
            assert x_s.shape[1] == 129, f"State dim wrong: {x_s.shape[1]} (expected 129 = 18×6 + 3 + 3)"
            assert y_s_n.shape[1] == 129, f"Target dim wrong: {y_s_n.shape[1]}"
            
            print(f"\n✅ All shape checks passed!")
            return True
        else:
            print("⚠️  Dataset is empty, please check data directories")
            return False
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass."""
    print("\n" + "="*80)
    print("Testing Model...")
    print("="*80)
    
    try:
        # Create dummy inputs
        batch_size = 4
        seq_len = 60
        input_imu_dim = 63  # 7 IMUs × 9
        state_dim = 129  # 18×6 + 3 + 3
        
        x_imu = torch.randn(batch_size, seq_len, input_imu_dim)
        x_s = torch.randn(batch_size, seq_len, state_dim)
        
        print(f"Input shapes:")
        print(f"   - x_imu: {x_imu.shape}")
        print(f"   - x_s: {x_s.shape}")
        
        # Create model
        model = TF_RNN_Past_State(
            input_size_imu=input_imu_dim,
            size_s=state_dim,
            rnn_hid_size=512,
            tf_hid_size=1024,
            tf_in_dim=256,
            n_heads=16,
            tf_layers=4,
            dropout=0.0,
            in_dropout=0.0,
            past_state_dropout=0.8,
            with_rnn=True,
            with_acc_sum=False
        )
        
        print(f"\n✅ Model created successfully")
        
        # Forward pass
        y_pred = model(x_imu, x_s)
        
        print(f"\nOutput shape: {y_pred.shape} (expected: [{batch_size}, {seq_len}, {state_dim}])")
        
        assert y_pred.shape == (batch_size, seq_len, state_dim), f"Output shape mismatch: {y_pred.shape}"
        
        print(f"✅ Forward pass successful!")
        
        # Test loss computation
        y_target = torch.randn_like(y_pred)
        
        # Flatten
        y_pred_flat = y_pred.reshape(-1, state_dim)
        y_target_flat = y_target.reshape(-1, state_dim)
        
        # Human part loss (rot + root_vel)
        human_dim = 18 * 6 + 3
        loss_human = loss_q_only_2axis(y_target_flat[:, :human_dim], y_pred_flat[:, :human_dim])
        
        # Object loss
        loss_obj = ((y_pred_flat[:, -3:] - y_target_flat[:, -3:]) ** 2).mean()
        
        # Jerk loss (only on rotation part: 18*6 = 108 dims)
        rot_dim = 18 * 6
        loss_j = loss_jerk(y_pred[:, :, :rot_dim])
        
        print(f"\nLoss computation:")
        print(f"   - Human loss: {loss_human.item():.6f}")
        print(f"   - Object loss: {loss_obj.item():.6f}")
        print(f"   - Jerk loss: {loss_j.item():.6f}" if loss_j is not None else "   - Jerk loss: None")
        
        total_loss = loss_human + loss_obj
        if loss_j is not None:
            total_loss += loss_j
        
        # Backward pass
        total_loss.backward()
        
        print(f"✅ Loss computation and backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test DataLoader."""
    print("\n" + "="*80)
    print("Testing DataLoader...")
    print("="*80)
    
    try:
        from torch.utils.data import DataLoader
        
        dataset = OMOMODatasetTIPFormat(
            data_dirs=['../../process/processed_data_OMOMO/train'],
            seq_len=60,
            use_object_imu=True,
            random_sample=True,
        )
        
        if len(dataset) == 0:
            print("⚠️  Dataset is empty, skipping DataLoader test")
            return False
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
            drop_last=False,
        )
        
        print(f"✅ DataLoader created successfully")
        print(f"   - Batch size: 4")
        print(f"   - Number of batches: {len(loader)}")
        
        # Get one batch
        x_imu, x_s, y_s_n = next(iter(loader))
        
        print(f"\nBatch shapes:")
        print(f"   - x_imu: {x_imu.shape}")
        print(f"   - x_s: {x_s.shape}")
        print(f"   - y_s_n: {y_s_n.shape}")
        
        assert x_imu.dim() == 3, "IMU should be 3D"
        assert x_s.dim() == 3, "State should be 3D"
        assert y_s_n.dim() == 3, "Target should be 3D"
        
        print(f"✅ DataLoader test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("TIP Format Pipeline Test")
    print("="*80 + "\n")
    
    results = []
    
    # Test dataset
    results.append(("Dataset", test_dataset()))
    
    # Test model
    results.append(("Model", test_model()))
    
    # Test dataloader
    results.append(("DataLoader", test_dataloader()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The TIP-format pipeline is ready to use.")
        print("\nTo start training, run:")
        print("   bash train_omomo_tip_format.sh")
        print("\nOr:")
        print("   python train_tip_format.py --cuda --use_object_imu")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

