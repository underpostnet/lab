import tensorflow as tf
import datetime

print(f"[{datetime.datetime.now()}] TensorFlow version: {tf.__version__}")

# List all physical GPUs
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    print(f"[{datetime.datetime.now()}] Found {len(physical_devices)} Physical GPU(s):")
    for gpu in physical_devices:
        print(f"[{datetime.datetime.now()}]   - {gpu}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print(f"[{datetime.datetime.now()}] No Physical GPUs found.")

# List all logical devices (including CPU and GPU)
logical_devices = tf.config.list_logical_devices()
print(f"[{datetime.datetime.now()}] Found {len(logical_devices)} Logical Device(s):")
for device in logical_devices:
    print(f"[{datetime.datetime.now()}]   - {device}")

# Simple test to verify GPU operation if available
if physical_devices:
    print(
        f"[{datetime.datetime.now()}] Running a simple TensorFlow operation on GPU..."
    )
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(
                f"[{datetime.datetime.now()}] Matrix multiplication result (should be on GPU):"
            )
            print(c.numpy())
            print(
                f"[{datetime.datetime.now()}] TensorFlow operation successfully executed on GPU."
            )
    except RuntimeError as e:
        print(
            f"[{datetime.datetime.now()}] Error running TensorFlow operation on GPU: {e}"
        )
else:
    print(
        f"[{datetime.datetime.now()}] No GPU found, skipping TensorFlow operation test on GPU."
    )

print(f"[{datetime.datetime.now()}] GPU check complete.")
