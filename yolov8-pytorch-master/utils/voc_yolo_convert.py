import xml.etree.ElementTree as ET
import os
import cv2
from xml.dom import minidom



# classes = []

# def convert(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = (box[0] + box[1]) / 2.0 - 1
#     y = (box[2] + box[3]) / 2.0 - 1
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)
#
#
# def convert_annotation(xmlpath, xmlname):
#     with open(xmlpath, "r", encoding='utf-8') as in_file:
#         txtname = xmlname[:-4] + '.txt'
#         txtfile = os.path.join(txtpath, txtname)
#         tree = ET.parse(in_file)
#         root = tree.getroot()
#         filename = root.find('filename')
#         img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
#         h, w = img.shape[:2]
#         res = []
#         for obj in root.iter('object'):
#             cls = obj.find('name').text
#             if cls not in classes:
#                 classes.append(cls)
#             cls_id = classes.index(cls)
#             xmlbox = obj.find('bndbox')
#             b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#                  float(xmlbox.find('ymax').text))
#             bb = convert((w, h), b)
#             res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
#         if len(res) != 0:
#             with open(txtfile, 'w+') as f:
#                 f.write('\n'.join(res))
#
#
# if __name__ == "__main__":
#     postfix = 'jpg'
#     imgpath = 'VOCdevkit/JPEGImages'
#     xmlpath = 'VOCdevkit/Annotations'
#     txtpath = 'VOCdevkit/txt'
#
#     if not os.path.exists(txtpath):
#         os.makedirs(txtpath, exist_ok=True)
#
#     list = os.listdir(xmlpath)
#     error_file_list = []
#     for i in range(0, len(list)):
#         try:
#             path = os.path.join(xmlpath, list[i])
#             if ('.xml' in path) or ('.XML' in path):
#                 convert_annotation(path, list[i])
#                 print(f'file {list[i]} convert success.')
#             else:
#                 print(f'file {list[i]} is not xml format.')
#         except Exception as e:
#             print(f'file {list[i]} convert error.')
#             print(f'error message:\n{e}')
#             error_file_list.append(list[i])
#     print(f'this file convert failure\n{error_file_list}')
#     print(f'Dataset Classes:{classes}')




def yolo_to_voc(txt_path, img_path, classes, output_xml_dir):
    """
    将YOLO TXT标注转换为VOC XML格式

    参数:
        txt_path: YOLO格式的TXT文件路径
        img_path: 对应的图片文件路径
        classes: 类别名称列表 (如 ['cat', 'dog'])
        output_xml_dir: XML输出目录
    """
    # 读取图片获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    height, width, _ = img.shape

    # 创建XML根节点
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.dirname(img_path)
    ET.SubElement(root, "filename").text = os.path.basename(img_path)

    # 图片尺寸信息
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"  # 假设RGB图像

    # 读取TXT文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # 跳过格式错误的行

        class_id, x_center, y_center, w, h = map(float, parts)
        class_name = classes[int(class_id)]

        # 将YOLO相对坐标转为VOC绝对坐标
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        xmin = int(x_center - w / 2)
        xmax = int(x_center + w / 2)
        ymin = int(y_center - h / 2)
        ymax = int(y_center + h / 2)

        # 创建object节点
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        # 添加bndbox
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    # 格式化输出XML
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 保存XML文件
    xml_filename = os.path.splitext(os.path.basename(txt_path))[0] + ".xml"
    os.makedirs(output_xml_dir, exist_ok=True)
    output_path = os.path.join(output_xml_dir, xml_filename)

    with open(output_path, 'w') as f:
        f.write(pretty_xml)

def classes_form(name_path):
    with open(name_path, 'r') as f:
        lines = [i.strip() for i in f.readlines()]

    return lines


if __name__ == "__main__":
    # 配置参数
    txt_dir = "D:\project\yolov8\easy_YOLOv8\YOLOv8lite\coco128\labels\\train2017"  # YOLO TXT文件目录
    img_dir = "D:\project\yolov8\easy_YOLOv8\YOLOv8lite\coco128\images\\train2017"  # 图片目录
    output_dir = "D:\project\yolov8\easy_YOLOv8\yolov8-pytorch-master\VOCdevkit\VOC2007\Annotations"  # 输出XML目录

    name_path = "D:\project\yolov8\easy_YOLOv8\yolov8-pytorch-master\model_data\coco_classes.txt"
    classes = classes_form(name_path)

    # 遍历转换所有TXT文件
    error_files = []
    for txt_name in os.listdir(txt_dir):
        if not txt_name.endswith('.txt'):
            continue

        try:
            base_name = os.path.splitext(txt_name)[0]
            txt_path = os.path.join(txt_dir, txt_name)
            img_path = os.path.join(img_dir, f"{base_name}.jpg")  # 假设图片为JPG格式

            yolo_to_voc(txt_path, img_path, classes, output_dir)
            print(f"转换成功: {txt_name} -> {base_name}.xml")
        except Exception as e:
            print(f"转换失败 {txt_name}: {str(e)}")
            error_files.append(txt_name)

    print(f"\n转换完成! 失败文件: {error_files}")