import shutil
import xml.etree.ElementTree as et
from pathlib import Path
from typing import List, Optional, NamedTuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow.keras.backend as K


DATA_DIR = '../../data/LADD_V4/winter_moscow_2018'
EFFICIENT_NET_SIZES = ((224,7), (240,7), (260,8), (300,9), (380,11), (456,14), (528,16), (600,18))


class Rectangle(NamedTuple):
    """Хранит координаты прямоугольника (xmin, ymin) - (xmax, ymax)"""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def w(self) -> int:
        """Ширина"""
        return self.xmax - self.xmin

    @property
    def h(self) -> int:
        """Высота"""
        return self.ymax - self.ymin

    @property
    def square(self) -> float:
        """Площадь"""
        return self.w * self.h

    def __repr(self) -> str:
        return f'Rectangle(x1={self.xmin},y1={self.ymin},x2={self.xmax},y2={self.ymax})'


class Annotation(NamedTuple):
    """Аннотация к изображению - bbox + класс объекта"""
    label: str
    bbox: Rectangle


class AnnotationFileReader:
    """Чтение файла с аннотациями из LADD (Pascal VOC)"""

    def __init__(self, filepath: str) -> None:
        self.filepath: Path = Path(filepath)

    def read_annotations(self) -> List[Annotation]:
        annotations = []
        root = et.parse(str(self.filepath)).getroot()
        for obj in root.iter('object'):
            bndbox = obj.find('bndbox')
            assert bndbox is not None
            annotation = Annotation(
                label=self._text(obj.find('name'), default=''),
                bbox=Rectangle(
                    xmin=int(self._text(bndbox.find('xmin'), default='0')),
                    ymin=int(self._text(bndbox.find('ymin'), default='0')),
                    xmax=int(self._text(bndbox.find('xmax'), default='0')),
                    ymax=int(self._text(bndbox.find('ymax'), default='0')),
                )
            )
            annotations.append(annotation)
        return annotations

    def _text(self, element: Optional[et.Element], default: str) -> str:
        if element is None:
            return default
        text = element.text
        if text is None:
            return default
        return text

    def __repr__(self) -> str:
        path = str(self.filepath)
        return f"AnnotationFile('{path}')"


def scale(src, x_factor, y_factor) -> Annotation:
    """Масштабирование координат"""
    return Annotation(
        label = src.label,
        bbox = Rectangle(
            xmin = round(src.bbox.xmin * x_factor),
            xmax = round(src.bbox.xmax * x_factor),
            ymin = round(src.bbox.ymin * y_factor),
            ymax = round(src.bbox.ymax * y_factor)
        )
    )


def shift(src, x_shift, y_shift) -> Annotation:
    """Сдвиг координат"""
    return Annotation(
        label = src.label,
        bbox = Rectangle(
            xmin = round(src.bbox.xmin - x_shift),
            xmax = round(src.bbox.xmax - x_shift),
            ymin = round(src.bbox.ymin - y_shift),
            ymax = round(src.bbox.ymax - y_shift)
        )
    )


def overlap_annotations(scaled_anns, left, top, right, bottom, CROP_SIZE=224) -> List:
    """Пересечение аннотаций с кропом изображения"""
    crop_anns = []
    for ann in scaled_anns:
        if ann.bbox.xmin >= left and ann.bbox.ymin >= top:
            if ann.bbox.xmax <= right and ann.bbox.ymax <= bottom:
                crop_anns.append(shift(ann, left, top))
            else:
                if ann.bbox.xmax - right < ann.bbox.w/3 and ann.bbox.ymax - bottom < ann.bbox.h/3:
                    crop_anns.append(Annotation(label=ann.label, bbox=Rectangle(
                        xmin=ann.bbox.xmin-left, ymin=ann.bbox.ymin-top, 
                        xmax=min(CROP_SIZE, ann.bbox.xmax-left), ymax=min(CROP_SIZE, ann.bbox.ymax-top))))
    return crop_anns


def ann_to_numpy(ann):
    """Convert coordinates according to tf bbox output: ymin, xmin, ymax, xmax, 1 - Pedestrian class """
    bb = ann.bbox
    return np.array((bb.xmin, bb.ymin, bb.xmax, bb.ymax))


def get_feature_map(bboxes, crop_size, fm_size):
    y = np.zeros((fm_size, fm_size), dtype=np.uint8)

    box_size = crop_size / fm_size
    
    for bb in bboxes:

        bx = int(np.floor( ( (bb[0]+bb[2]-1)/2) / box_size))
        by = int(np.floor( ( (bb[1]+bb[3]-1)/2) / box_size))
        try:
            y [by, bx] = 1
        except IndexError:
             print(bboxes)
             print(nn_crop_size)

    return  y


def crop_sample(idx, CROP_SIZE, FEATURE_MAP_SIZE):
    """Crop image for WxH crops and resize every crop to CROP_SIZE 
    return N crops with annotations"""
    N = 4
    W, H = 4, 3


    img_path = DATA_DIR + '/JPEGImages/' + f'{idx}.jpg'
    img = load_img(img_path)
    ann_path = DATA_DIR + '/Annotations/' + f'{idx}.xml'
    anns = AnnotationFileReader(ann_path).read_annotations()

    img_r = img.resize(size=(W*CROP_SIZE, H*CROP_SIZE))

    k_x = W*CROP_SIZE / img.width
    k_y = H*CROP_SIZE / img.height

    scaled_anns = [scale(a, k_x, k_y) for a in anns]
    out = []

    for w in range(W):
        for h in range(H):
            left = w * CROP_SIZE
            top  = h * CROP_SIZE
            right  = (w+1) * CROP_SIZE
            bottom = (h+1) * CROP_SIZE
            
            crop_img = img_r.crop((left, top, right, bottom))
            crop_anns = overlap_annotations(scaled_anns, left, top, right, bottom, CROP_SIZE)
            crop_bboxes = [ann_to_numpy(ann) for ann in crop_anns]

            y = get_feature_map(crop_bboxes, CROP_SIZE, FEATURE_MAP_SIZE)
            # print(y.sum())
            out.append(
                ((w,h),
                img_to_array(crop_img, dtype=np.uint8),
                y,
                crop_anns
                )
            )

    return sorted(out, key=lambda x: x[2].sum(), reverse=True)[:N]


@tf.function
def focal_loss(
    y_true,
    y_pred,
    alpha = 0.25,
    gamma = 2.0,
    from_logits = False,
) -> tf.Tensor:
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce)


