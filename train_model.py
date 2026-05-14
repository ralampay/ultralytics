from ultralytics import YOLO

def main():
    model = YOLO("ultralytics/cfg/models/ext/doubleconv-yolo.yaml")

    results = model.train(
        data="skyfusion-dataset/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/final_project",
        name="doubleconv_exp",
        plots=True,
    )

if __name__ == '__main__':
    main()