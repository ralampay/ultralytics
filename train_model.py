from ultralytics import YOLO

def main():
    model = YOLO("ultralytics/cfg/models/ext/doubleconv-yolo.yaml")

    results = model.train(
        data="skyfusion.v1i.yolov11/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/final_project",
        name="doubleconv_exp",
        plots=True,
    )

if __name__ == '__main__':
    main()