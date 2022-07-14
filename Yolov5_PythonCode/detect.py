# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import base64
import io

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(reda_img,
        weights=ROOT / 'runs/train/exp/weights/best.pt',  # model.pt path(s)
        source=ROOT / 'hassanRabat.jpg',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/custom_data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    global label
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    #     bs = len(dataset)  # batch_size
    # else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt,reda_img=reda_img)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    #REDA CODE
                    RGB_img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    pil_img=Image.fromarray(RGB_img)
                    buff=io.BytesIO()
                    pil_img.save(buff,format="JPEG")
                    img_str=base64.b64encode(buff.getvalue())
                    return ""+str(img_str,'utf-8')
                    #END
                #     cv2.imwrite(save_path, im0)
                # else:  # 'video' or 'stream'
                #     if vid_path[i] != save_path:  # new video
                #         vid_path[i] = save_path
                #         if isinstance(vid_writer[i], cv2.VideoWriter):
                #             vid_writer[i].release()  # release previous video writer
                #         if vid_cap:  # video
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         else:  # stream
                #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #     vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'hassanRabat.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/custom_data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(reda_img):
    decoded_data=base64.b64decode(reda_img)
    np_data=np.fromstring(decoded_data,np.uint8)
    reda_img=cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    RGB_img = cv2.cvtColor(reda_img, cv2.COLOR_BGR2RGB)
    opt = parse_opt()
    check_requirements(exclude=('tensorboard', 'thop'))
    stringImg=run(**vars(opt),reda_img=reda_img)
    return stringImg

def MonumentLabel():
    monuments=['Bab El-khamis', 'Bab Mansour', 'Bab Berdaine',
               'Bab chellah', 'Grande mosquee de Meknes',
               'Heri es Souani', 'Koutoubia', 'Medersa Attarine',
               'Menara', 'Mosquee Hassan2', 'Musee Mohammed VI',
               'Musee Nejjarine', 'Oualili', 'Palais El Badi',
               'Palais Royal de Fes', 'Porte Bab Boujloud',
               'Tannerie Chouara', 'Tombeaux Saadiens',
               'Tour Hassan']

    infos=['This is the Thursday Gate of Marrakech. It is an original Almoravid gate. It was reconstructed in early 1800’s. It’s curved entrance is decorated in intricate pattern. There are merlons on top this arched gateway. It is located in the northernmost part of the Medina. This gate has interesting history. Most parts of the Marrakech Medina are famous for it’s markets. This gate is famous for it’s flea market which used to be held on Thursday. This market is still held here everyday here, except Friday. The biggest one is still organised on Thursdays.',
           'Having decided to make the town of Meknès the capital of his kingdom, Mulay Isma\'il dedicated himself throughout his 45-year reign to the construction of 40 km of bastions and walls, monumental gateways, granaries, enormous stables, gardens and large lakes.Bab Mansur al-\'Alaj, which is one of Mulay Isma\'il\'s last projects, is the most famous of the 20 gateways that provide access to the royal city.',
           'This gate to the imperial city of Meknes was built in the seventeenth century by Sultan Moulay Ismail. This gate opens into the northern side of the Medina. It’s name is derived from the market of packsaddles which used to exist in this section of the town. This gate lies next to the mosque which is it’s namesake. It is humongous gate with square towers. It is covered with beautiful green zellij tiles. It is true prototype of Saadian architecture.',
           'Chellah was officially built by the Romans around 40 AD and was known at the time as Sala Colonia, or Sala. This major port city included key Roman architectural elements such as a forum, a triumphal arch, a temple, aqueducts, and a principal roadway.',
           'The Grand Mosque of Meknes is the historic main mosque of the old city of Meknes, Morocco. It is the largest and most important mosque in the old city and one of its oldest monuments.',
           'The Heri es-Swani or Heri es-Souani, sometimes also transliterated as Hury as-Swani or Hri Swani, is a historic monument in Meknes, Morocco. It was a massive structure that served as a granary or silo for the Imperial Kasbah built by Moulay Isma\'il during his reign from 1672 to 1727.',
           'The Kutubiyya Mosque or Koutoubia Mosque is the largest mosque in Marrakesh, Morocco. The mosque\'s name is also variably rendered as Jami\' al-Kutubiyah, Kutubiya Mosque, Kutubiyyin Mosque, and Mosque of the Booksellers.',
           'The Al-Attarine Madrasa or Medersa al-Attarine is a madrasa in Fes, Morocco, near the Al-Qarawiyyin Mosque. It was built by the Marinid sultan Uthman II Abu Said in 1323-5. The madrasa takes its name from the Souk al-Attarine, the spice and perfume market.',
           'The Menara Gardens are Marrakech’s most famous gardens. They were established during the twelfth century around a lake, which was used to water the fruit and vegetables planted in the grounds. It was initially commissioned by Abd al-Mu’min, leader of the Almohad Movement. Later, the gardens were renovated in 1870.',
           'The Hassan II Mosque is a mosque in Casablanca, Morocco. It is the second largest functioning mosque in Africa and is the 7th largest in the world. Its minaret is the world\'s second tallest minaret at 210 metres.',
           'The Mohammed VI Museum of Modern and Contemporary Art, abbreviated MMVI, is a contemporary and modern art museum in Rabat, Morocco which opened in 2014. It is one of fourteen museums of the National Foundation of Museums of Morocco. The museum curates modern and contemporary Moroccan and international art.',
           'Funduq al-Najjarin is a historic funduq in Fes el Bali, the old medina quarter in the city of Fez, Morocco. The funduq is situated in the heart of the medina, at Al-Najjarin Square, which is also notable for the Nejjarine Fountain, an attached saqayya or traditional public fountain.',
           'Oualili is a partly excavated Berber-Roman city in Morocco situated near the city of Meknes, and may have been the capital of the kingdom of Mauretania, at least from the time of King Juba II.',
           'El Badi Palace or Badi Palace is a ruined palace located in Marrakesh, Morocco. It was commissioned by the sultan Ahmad al-Mansur of the Saadian dynasty a few months after his accession in 1578, with construction and embellishment continuing throughout most of his reign.',
           'The Dar al-Makhzen or Royal Palace of Fez is the royal palace of the King of Morocco in the city of Fez, Morocco. Its original foundation dates back to the foundation of Fes el-Jdid, the royal citadel of the Marinid dynasty, in 1276 CE. Most of the palace today dates from the Alaouite era.',
           'Bab Bou Jeloud is an ornate city gate in Fes el Bali, the old city of Fez, Morocco. The current gate dates was built by the French colonial administration in 1913 to serve as the grand entrance to the old city.',
           'Chouara Tannery is one of the three tanneries in the city of Fez, Morocco. It is the largest tannery in the city and one of the oldest. It is located in the Fes el Bali, the oldest medina quarter of the city, near the Saffarin Madrasa along the Oued Fes (also known as the Oued Bou Khrareb)',
           'The Saadian Tombs are a historic royal necropolis in Marrakesh, Morocco, located on the south side of the Kasbah Mosque, inside the royal kasbah district of the city.',
           'Hassan Tower or Tour Hassan is the minaret of an incomplete mosque in Rabat, Morocco. It was commissioned by Abu Yusuf Yaqub al-Mansur, the third Caliph of the Almohad Caliphate, near the end of the 12th century.']

    monData={}
    for i in range(len(monuments)):
        monData[monuments[i]]=infos[i]

    return monData[label[:-5]]

def monumentName():
    return label[:-5]
# main()