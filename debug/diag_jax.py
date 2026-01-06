import os
import sys

print(f"Python version: {sys.version}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

try:
    import jax
    print(f"JAX version: {jax.__version__}")

    # Try to get devices
    try:
        devices = jax.devices()
        print(f"Devices: {devices}")
    except Exception as dev_err:
        print(f"Error getting devices: {dev_err}")

    # Check backends
    try:
        import jax.extend as jex
        backend = jex.backend.get_backend()
        print(f"Default backend: {backend.platform}")
    except Exception as back_err:
        print(f"Error getting backend: {back_err}")

except Exception as e:
    print(f"Main Error: {e}")
    import traceback
    traceback.print_exc()
