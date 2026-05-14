from ultralytics import YOLO
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

try:
    model = YOLO("ultralytics/cfg/models/ext/doubleconv-yolo.yaml")
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model {e}")
    exit()

x = torch.randn(1, 3, 640, 640)
y = model.model(x)
if isinstance(y, torch.Tensor):
    print(y.shape)
elif isinstance(y, (list, tuple)):
    print([getattr(t, "shape", type(t).__name__) for t in y])
else:
    print(type(y).__name__)
print(model.model)

# print model summmary 
print("\n=====Model Summary=====")
model.info()

# run dummy forward pass 
print("\n======Testing Dummy Img=====")
try:
    dummy_input = torch.randn(1, 3, 640, 640) #fake image
    output = model.model(dummy_input)
    print(f"Forward pass is successful. Output type: {type(output)}")
except Exception as e:
    print(f"Error during forward pass {e}")

# confirm DoubleConvBackbone appears
print("\n======Confirming DoubleConvBackbone=====")
for name, module in model.model.named_modules():
    if "DoubleConvBackbone" in str(type(module)):
        print(f"DoubleConvBackbone found. Layer name: {name} is a DoubleConvBackbone")