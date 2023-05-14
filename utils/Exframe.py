from pathlib import Path, PurePath
from typing import Union
import time
import cv2
import os

class ExtractImg(object):
    def __init__(self, videopath:Path, savepath:Path, delay: int=40) -> None:
        self.spath = savepath
        self.vpath = videopath
        self.delay = delay
        cv2.namedWindow("cv", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cv", 640, 480)
        self.cap = cv2.VideoCapture(str(self.vpath))
        self._timeflag = 0
        if not savepath.exists():
            os.mkdir("./frameSave")

    def _videoPlay(self,size:list) -> None:
        self.cap.set(3, size[0])
        self.cap.set(4, size[1])
        # print(self.cap.isOpened())
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("cv", frame)
            if cv2.waitKey(self.delay) & 0xFF == ord('c'):
                cv2.imwrite(str(PurePath.joinpath(self.spath,
                          "{}.jpg".format(str(time.time())))), frame)
                print("sucsess")
                time.sleep(1)
            elif cv2.waitKey(self.delay) & 0xFF == 27:
                break

    def ExtractAll(self) -> None:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("cv", frame)
            cv2.imwrite(str(PurePath.joinpath(self.spath,
                      "{}.jpg".format(self.timeName()))),frame)
            print("保存成功")
            # time.sleep(1)
            if cv2.waitKey(self.delay) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def CutVideo(self) -> None:
        ifm = input("文件中已经存在{}张图片，是否有继续添加"
                    "(Y or N)：".format(len(os.listdir(self.spath))))
        if self.spath.exists() and ifm=='Y':
            self._videoPlay(size=[640, 480])
        elif self.spath.exists() and ifm == 'N':
            return None
        else:
            print("\n请输入Y（yes）或者N（no）")
        cv2.destroyAllWindows()
        self.cap.release()

    @staticmethod
    def statistics(path: Union[str, Path], count: int = 5305) -> None:
        assert isinstance(path,(Path, str)), print("请输入的路径")
        l = os.listdir(str(path))
        # print(l)
        print("存在图片{}张：".format(len(l)))
        for i, img in enumerate(l):
            src = PurePath.joinpath(path,img)
            dst = PurePath.joinpath(path, Path(str(count+i)).with_suffix(".jpg"))
            os.rename(src,dst)


if __name__ == "__main__":
    # videopath = Path("vid.mp4")
    videopath = Path("sample.mp4")
    savepath = Path("frameSave")
    # savepath = Path("frameSave")
    a = ExtractImg(videopath=videopath, savepath=savepath)
    # a.ExtractAll()
    a.CutVideo()
    # a.statistics(savepath)

