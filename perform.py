import sys
import os
import time
from pathlib import Path
from typing import Union
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO('yolov8l.yaml')
    # model = YOLO('yolov8m.pt')
    
    # 做预训练
    # model = YOLO('yolov8x.pt')
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # Train the model
    # model.train(data="HSTS6.yaml", epochs=150, imgsz=640)
    model.train(data="VOC.yaml", epochs=150, imgsz=640)


def onnx():
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('/home/you/Desktop/YOLOv8/easy_YOLOv8/runs/detect/m6/weights/best.pt')  # load a custom trained

    # Export the model
    model.export(format='onnx')


def test_img():
    model = YOLO("/home/you/Desktop/YOLOv8/easy_YOLOv8/runs/detect/m6/weights/best.pt")
    img = cv2.imread("/home/you/Desktop/YOLOv8/easy_YOLOv8/7.jpg")
    res = model(img)
    ann = res[0].plot()
    while True:
        cv2.imshow("yolo", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cur_path = sys.path[0]
    print(cur_path, sys.path)

    if os.path.exists(cur_path):
        cv2.imwrite(cur_path + "out2.jpg", ann)
    else:
        os.mkdir(cur_path)
        cv2.imwrite(cur_path + "out2.jpg", ann)


def test_video():
    # model = YOLO("/home/you/Desktop/YOLOv8/easy_YOLOv8/runs/detect/train5/weights/best.pt")
    model = YOLO("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/Faster-YOLOv8l/weights/best.pt")
    # pa = "/home/you/Desktop/tools/videoSet/speedboat1.mp4"
    pa = "/home/youtian/Documents/pro/2023/2023海上高速目标检测/videoSet/airplane2.mp4"
    cap = cv2.VideoCapture(pa)
    size = (int(cap .get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap .get(cv2.CAP_PROP_FRAME_HEIGHT)),)

    out = cv2.VideoWriter('airplane.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            res = model(frame)
            ann = res[0].plot()
            cv2.imshow("yolo", ann)
            out.write(ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()


def test_folders(model_path: str = "/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/YOLOv8l/weights/best.pt",
                 srcpath: str = "/home/youtian/Documents/pro/pyCode/easy_YOLOv8/resource") -> None:
    model = YOLO(model_path)
    srcpath = Path(srcpath) if not isinstance(srcpath, Path) else srcpath
    dstpath = sys.path[0]
    dstpath = Path(dstpath) / Path(str(time.time())[-10:])
    os.mkdir(dstpath)
    for index, i in enumerate(srcpath.iterdir()):
        res = model(cv2.imread(str(i)))
        ann = res[0].plot()
        # 把测试的图片提前resize成相同的size
        ann = cv2.resize(ann, (640, 640))
        cv2.imwrite(str(Path(dstpath) / Path("{}.jpg".format(index))), ann)


def tracker():
    pa = "/home/you/Downloads/l.mp4"
    cap = cv2.VideoCapture(pa)
    size = (int(cap .get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap .get(cv2.CAP_PROP_FRAME_HEIGHT)),)
    # cap = cv2.VideoCapture("/home/you/Desktop/YOLOv8/easy_YOLOv8//sample.mp4")
    model = YOLO("/home/you/Desktop/YOLOv8/easy_YOLOv8/runs/detect/train3"
                 "/weights/best.pt")
    flag = 0
    out = cv2.VideoWriter('airplaneSort.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)
    while True:
        if flag<1:
            flag += 1
            continue
        else:
            flag += 1
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            # result.boxes.id.cpu().numpy().astype(int)
            try:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Id {id}",
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
            except Exception as e:
                print(e)
            cv2.imshow("frame", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # # # or a segmentation model .i.e yolov8n-seg.pt
    # # model.track(
    # #     source="vid.mp4",
    # #     stream=True,
    # #     tracker="botsort.yaml",  # or 'bytetrack.yaml'
    # #     show=True,
    # # )
    #
    # for result in model.track(source="vid.mp4"):
    #     print(
    #         result.boxes.id.cpu().numpy().astype(int)
    #     )


if __name__ == "__main__":
    # train()
    test_video()
    # test_folders()
    # test_img()
    # tracker()
    # onnx()

    # netron.start("/YOLOv8/runs/detect/m6re/weights/best.onnx")
