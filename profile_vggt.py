import torch
import nvtx
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Setup
device = "cuda"
dtype = torch.bfloat16 # A40 supports bfloat16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

# Load a small batch of images (e.g., 5-10 images to avoid OOM initially)
image_dir = "examples/chair"
image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.JPG', '.jpeg'))]
images = load_and_preprocess_images(image_paths).to(device)

def run_inference():
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # We profile the 'aggregator' because that's where aggregation.py is called
            with nvtx.annotate("VGGT_Aggregator", color="blue"):
                predictions = model(images)
    return predictions

# Warmup (important for accurate profiling)
print("Warming up...")
for _ in range(3):
    _ = run_inference()

# Actual Profiled Run
print("Starting profiled run...")
torch.cuda.synchronize()
run_inference()
torch.cuda.synchronize()