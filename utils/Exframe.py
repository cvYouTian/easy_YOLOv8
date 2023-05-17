# -*- coding: utf-8 -*-
# @Author  : cvYouTian
# @Software: PyCharm

from pathlib import Path, PurePath
from typing import Union
from tqdm import tqdm
import time
import cv2
import os


class ExtractImg(object):
    def __init__(self, videopath: Path, savepath: Path, delay: int = 1) -> None:
        self.spath = savepath
        self.vpath = videopath
        self.delay = delay
        cv2.namedWindow("cv", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cv", 640, 480)
        self.cap = cv2.VideoCapture(str(self.vpath))
        self._timeflag = 0
        if not savepath.exists():
            os.mkdir(Path(savepath))

    def _videoPlay(self, size: list) -> None:
        self.cap.set(3, size[0])
        self.cap.set(4, size[1])
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("cv", frame)
            if cv2.waitKey(self.delay) & 0xFF == ord('c'):
                cv2.imwrite(str(PurePath.joinpath(self.spath,
                                                  "{}.jpg".format(str(time.time())))), frame)
                print("保存成功")
                time.sleep(1)
            elif cv2.waitKey(self.delay) & 0xFF == 27:
                break

    def ExtractAll(self, frameGap: int=3) -> None:
        """
        这是将视频流中的帧全部抽出
        :frame: 跳帧
        :return:
        """
        while self.cap.isOpened():
            self._timeflag += 1

            ret, frame = self.cap.read()
            if ret:
                cv2.imshow("cv", frame)
                if self._timeflag % frameGap == 0:
                    cv2.imwrite(str(PurePath.joinpath(self.spath,
                                                      "{}.jpg".format(str(time.time())))), frame)
                    print("保存成功")
            if (cv2.waitKey(self.delay) & 0xFF == 27) or not ret:
                break
        cv2.destroyAllWindows()
        self.cap.release()
        self._timeflag = 0

    def CutVideo(self) -> None:
        """
        这是手动抽帧
        :return:
        """
        ifm = input("文件中已经存在{}张图片，是否有继续添加"
                    "(Y or N)：".format(len(os.listdir(self.spath))))
        if self.spath.exists() and ifm == 'Y':
            self._videoPlay(size=[640, 480])
        elif self.spath.exists() and ifm == 'N':
            return None
        else:
            print("\n请输入Y（yes）或者N（no）")
        cv2.destroyAllWindows()
        self.cap.release()

    @staticmethod
    def statistics(path: Union[str, Path], count: int = 5305) -> None:
        """
        这是存放图片的文件夹安升序重命名
        :param path:需要重命名的文件文件
        :param count:观察图片总数添加使用
        :return:
        """
        assert isinstance(path, (Path, str)), print("请输入的路径")
        l = os.listdir(str(path))
        # print(l)
        print("存在文件{}张！".format(len(l)))
        # 将保存图片文件中的图片按照升序的方法重命名
        for i, file in tqdm(enumerate(l)):
            suffix = Path(file).suffix
            src = PurePath.joinpath(path, file)
            dst = PurePath.joinpath(path, Path(str(count + i)).with_suffix(suffix))
            os.rename(src, dst)

    @staticmethod
    def choosen(xmlsrc: Union[str, Path], imgsrc: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        这是将xml文件夹对应的图片挑出来
        :param xmlsrc:目标xml文件
        :param imgsrc:frameImg文件
        :param dst:根据xml挑选出的img文件
        :return: None
        """
        l = os.listdir(str(xmlsrc))
        for i in tqdm(l):
            img = str(Path(i).with_suffix(".jpg"))
            if img in os.listdir(str(imgsrc)):
                os.system("cp ./{} ./{}".format(str(PurePath.joinpath(imgsrc, img)), str(dst)))


if __name__ == "__main__":
    # 目标视频文件
    videopath = Path("sample.mp4")
    # 图片保存文件
    savepath = Path("frameSave")
    # savepath = Path("frameSave")
    # 目标xml文件
    xmlpath = Path("ann")
    # 根据xml挑选出的img文件
    xin = Path("img")
    # 实例化
    a = ExtractImg(videopath=videopath, savepath=savepath)



    # 将帧全部抽出
    a.ExtractAll(frameGap=1)
    # 手动抽帧
    # a.CutVideo()
    # 根据xml文
    # a.choosen(xmlsrc=xmlpath, imgsrc=savepath, dst=xin)
    # a.statistics(xin)

