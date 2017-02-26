import numpy as np
import cv2
import xml.etree.ElementTree as et
import time


class Interface:
    def __init__(self, title, CAM_CODE, FLIP_CODE=False, XML_FILE="haarcascades/data.xml", SAVE_IMGS=False,
                 SAVE_DIRECTORY=None):
        # type: (object, object) -> object
        # Setting the hlines and vlines

        self.x1 = 0
        self.x2 = None
        self.y1 = 0
        self.y2 = None

        self.shape = None

        self.title = title
        self.save_imgs = SAVE_IMGS
        self.FLIP_CODE = FLIP_CODE
        self.i = 0
        self.CAM_CODE = CAM_CODE
        self.cam = cv2.VideoCapture(self.CAM_CODE)

        _, test = self.cam.read()
        self.lines, self.cols, _ = test.shape
        self.x2 = self.cols
        self.y2 = self.lines
        del test
        # To test the dimensions of the array
        # _, im = self.cam.read()
        # print im.shape
        # Prints ->> (480, 640, 3)
        #             lines cols  channels
        self.cascade_files_list, self.color_list = self.get_cascade_list(XML_FILE)  # default: "haarcascades/data.xml")
        self.cascade_classifier_list = self.get_cascade_classifier_list(self.cascade_files_list)

        #  def defineresources(self):
        # Defining my resources, cam recorder, cascade files list, color list, and cascade classifiers list.
        #      global i, cam, cascade_files_list, color_list, cascade_classifier_list
        #     i = 0

    def etiquette(self, name, mat, color, origin, xsize, ysize):
        cv2.rectangle(mat, origin, (origin[0] + xsize, origin[1] + ysize), color, 2)
        cv2.putText(mat, name, (origin[0], origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def detectcascade(self, img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.4, minNeighbors=3, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    def draw_rects(self, img, rects, color, etiqname):  # , imgname):
        for x1, y1, x2, y2 in rects:
            # if self.SAVE_DIRECTORY is not None:
            #    cv2.imwrite(self.SAVE_DIRECTORY + "/images/%s.jpg" % imgname, img[x1:x2, y1:y2])
            #   self.i += 1
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            self.etiquette(etiqname, img, color, (x1, y1), x2 - x1, y2 - y1)

    def get_cascade_list(self, xml_file):
        cascade_files_list = []
        _color_list = []
        tree = et.parse(xml_file)
        root = tree.getroot()
        i = 0
        while i < len(root):
            cascade_files_list.append([root[i][0].text, root[i][1].text, root[i][2].text])
            _color_list.append((int(root[i][3].text), int(root[i][4].text), int(root[i][5].text)))
            i += 1
        return cascade_files_list, _color_list

    def get_cascade_classifier_list(self, _cascade_files_list):
        _cascadeclassifier_list = []
        for elm in _cascade_files_list:
            _cascadeclassifier_list.append(cv2.CascadeClassifier(elm[0]))

        return _cascadeclassifier_list

    def draw_str(self, dst, target, s):
        x, y = target
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=2)
        cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    def detectndraw(self):
        global img_panel, cam, cascade_files_list, color_list, cascade_classifier_list

        while 1:
            ret, orig_img = self.cam.read()
            if self.FLIP_CODE:
                cv2.flip(src=img, dst=img, flipCode=1)

            # Specifying ROI: the Region of Interest
            # img = np.zeros((self.lines, self.cols, 3))
            img = orig_img[min(self.y1, self.y2): max(self.y1, self.y2),
                  min(self.x1, self.x2): max(self.x1, self.x2)]

            # orig_img = np.zeros((self.lines, self.cols, 3))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = cv2.equalizeHist(gray)

            N = len(self.cascade_classifier_list)
            j = 0
            while j < N:
                rects = self.detectcascade(gray, self.cascade_classifier_list[j])
                # draw_rects(img, rects, , draw_etiq=True)
                self.draw_rects(img=img, rects=rects, color=self.color_list[j], etiqname=self.cascade_files_list[j][1])
                # imgname=time.ctime())  #, draw_etq=True)
                j += 1

            orig_img[min(self.y1, self.y2): max(self.y1, self.y2),
            min(self.x1, self.x2): max(self.x1, self.x2)] = img

            if self.x1 is not None:
                self.draw_vline(orig_img, self.x1, (0, 100, 0))

            if self.x2 is not None:
                self.draw_vline(orig_img, self.x2, (0, 100, 0))

            if self.y1 is not None:
                self.draw_hline(orig_img, self.y1, (200, 0, 0))

            if self.y2 is not None:
                self.draw_hline(orig_img, self.y2, (200, 0, 0))

            self.draw_str(orig_img, (20, 20), time.ctime())
            cv2.imshow(self.title, orig_img)

            if 0xFF & cv2.waitKey(5) == 27:
                break

    def draw_vline(self, img, x, color):
        lines, cols, _ = img.shape
        cv2.line(img=img, pt1=(x, 0), pt2=(x, lines), color=color, thickness=2)

    def draw_hline(self, img, y, color):
        lines, cols, _ = img.shape
        cv2.line(img=img, pt1=(0, y), pt2=(cols, y), color=color, thickness=2)

    def __del__(self):
        cv2.destroyAllWindows()


class Controller:
    def __init__(self, interface):
        # x1=None, x2=None, y1=None, y2=None
        self.interface = interface

        # self.set_interface_xy(x1, x2, y1, y2)

        cv2.namedWindow(interface.title)

        # To test the dimensions of the array
        # _, im = self.cam.read()
        # print im.shape
        # Prints ->> (480, 640, 3)
        #             lines cols  channels

        cv2.createTrackbar("Vertical 1", interface.title, 0, self.interface.cols, self.vline1)
        cv2.createTrackbar("Vertical 2", interface.title, 0, self.interface.cols, self.vline2)

        cv2.createTrackbar("Horizontal 1", interface.title, 0, self.interface.lines, self.hline1)
        cv2.createTrackbar("Horizontal 2", interface.title, 0, self.interface.lines, self.hline2)

    def set_interface_xy(self, x1, x2, y1, y2):
        self.interface.x1 = x1
        self.interface.x2 = x2
        self.interface.y1 = y1
        self.interface.y2 = y2

    def vline1(self, x):
        self.interface.x1 = x

    def vline2(self, x):
        self.interface.x2 = x

    def hline1(self, y):
        self.interface.y1 = y

    def hline2(self, y):
        self.interface.y2 = y

    def __del__(self):
        cv2.destroyAllWindows()
