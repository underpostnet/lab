import os
import tensorflow as tf


# Set XLA_FLAGS to point to the CUDA data directory where libdevice is located.
# This must be set BEFORE importing tensorflow to ensure it takes effect.
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/root/.conda/envs/cuda_env"

print("--- Starting GPU and Library Check ---")

# --- Check for GPU devices ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU if multiple are detected.
        # This must be set before TensorFlow has initialized GPUs.
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(
            f"TensorFlow detected {len(gpus)} Physical GPUs, "
            f"{len(logical_gpus)} Logical GPUs. Using: {gpus[0].name}"
        )
    except RuntimeError as e:
        # Catch potential errors if visible devices are set after GPU initialization
        print(f"RuntimeError during GPU configuration: {e}")
else:
    print("TensorFlow did not detect any GPU devices. Running on CPU.")

# --- Environment Variable Check Accelerated Linear Algebra ---
xla_flags_env = os.environ.get("XLA_FLAGS")
print(f"XLA_FLAGS environment variable (inside script): {xla_flags_env}")

tf_xla_flags_env = os.environ.get("TF_XLA_FLAGS")
print(f"TF_XLA_FLAGS environment variable (inside script): {tf_xla_flags_env}")

# --- Verify TensorFlow version and CUDA/cuDNN status (optional but helpful) ---
print(f"TensorFlow version: {tf.__version__}")
print(f"Is TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Is GPU available: {tf.test.is_gpu_available(cuda_only=False)}")

# --- Final confirmation ---
print("TensorFlow is configured to attempt running on GPU if available.")
print("--- GPU and Library Check Complete ---")
