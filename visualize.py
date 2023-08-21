from __future__ import division
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from dotenv import load_dotenv

import cv2, pymongo, os, certifi
import numpy as np

load_dotenv()

myclient = pymongo.MongoClient(os.getenv("DB_URL"),tlsCAFile=certifi.where())
mydb = myclient[os.getenv("DB")]
mycol = mydb[os.getenv("COLLECTION")]

# delete all previous output images - modified by Batu
for file in os.listdir('output'):
    os.remove(os.path.join('output', file))

def visualize_box_mask(im, results, labels, threshold=0.5):
    """
    Args:
        im (str/np.ndarray): path of image/np.ndarray read by cv2
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): Threshold of score.
    Returns:
        im (PIL.Image.Image): visualized image
    """
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    else:
        im = Image.fromarray(im)
    if 'masks' in results and 'boxes' in results:
        im = draw_mask(
            im, results['boxes'], results['masks'], labels, threshold=threshold)
    if 'boxes' in results:
        im = draw_box(im, results['boxes'], labels, threshold=threshold)
    if 'segm' in results:
        im = draw_segm(
            im,
            results['segm'],
            results['label'],
            results['score'],
            labels,
            threshold=threshold)
    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_mask(im, np_boxes, np_masks, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
            matix element:[class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of mask
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_color_map_list(len(labels))
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype('float32')
    clsid2color = {}
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]
    np_masks = np_masks[expect_boxes, :, :]
    for i in range(len(np_masks)):
        clsid, score = int(np_boxes[i][0]), np_boxes[i][1]
        mask = np_masks[i]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype('uint8'))

# function to find the fruit information from mongodb - modified by Batu
def find_the_fruit(fruit_name): 
    myquery = { "fruit": { "$regex": "^"+fruit_name } }
    mydoc = mycol.find(myquery)
    result = []
    for x in mydoc:
        result.append(x)
    result.sort(key=lambda x: x['price'])   # sort by price
    price = result[0]['price']
    provider = result[0]['provider']

    return '{}:{}/{} at {}'.format(fruit_name,price,'kg',provider)


def draw_box(im, np_boxes, labels, threshold=0.5):  # draw label - modified by Batu
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    if np_boxes.any():
        for dt in np_boxes:
            clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
            xmin, ymin, xmax, ymax = bbox
            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                'right_bottom:[{:.2f},{:.2f}]'.format(
                    int(clsid), score, xmin, ymin, xmax, ymax))
            w = xmax - xmin
            h = ymax - ymin
            if clsid not in clsid2color:
                clsid2color[clsid] = color_list[clsid]
            color = tuple(clsid2color[clsid])

            # draw bbox
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                (xmin, ymin)],
                width=draw_thickness,
                fill=color)

            # text = "{} {:.4f}".format(labels[clsid], score)   # draw name and confidence
            text = find_the_fruit(labels[clsid])
            
            for file in os.listdir('images'):   # remove input image
                os.remove(os.path.join('images', file))
            print('检测到：'+labels[clsid])
            font = ImageFont.truetype('得意黑.otf', 70)
            tw, th = draw.textsize(text)
            # draw.rectangle([(xmin, ymin), (tw, th)], fill=color)
            draw.text((300, ymin), text, font=font, fill=(255, 255, 255))

    else:
        # os.system('python3 result_window.py')
        x, y = im.size
        font = ImageFont.truetype('得意黑.otf', 70)
        text = 'NO FRUIT DETECTED'
        # drawing text size
        draw.text((x//2, y//2), text, font=font, align ="center", fill=(255, 255, 255))
 
    return im


def draw_segm(im,
              np_segms,
              np_label,
              np_score,
              labels,
              threshold=0.5,
              alpha=0.7):
    """
    Draw segmentation on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = get_color_map_list(len(labels))
    im = np.array(im).astype('float32')
    clsid2color = {}
    np_segms = np_segms.astype(np.uint8)
    for i in range(np_segms.shape[0]):
        mask, score, clsid = np_segms[i], np_score[i], np_label[i]
        if score < threshold:
            continue

        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
        sum_x = np.sum(mask, axis=0)
        x = np.where(sum_x > 0.5)[0]
        sum_y = np.sum(mask, axis=1)
        y = np.where(sum_y > 0.5)[0]
        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        cv2.rectangle(im, (x0, y0), (x1, y1),
                      tuple(color_mask.astype('int32').tolist()), 1)
        bbox_text = '%s %.2f' % (labels[clsid], score)
        t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
        cv2.rectangle(im, (x0, y0), (x0 + t_size[0], y0 - t_size[1] - 3),
                      tuple(color_mask.astype('int32').tolist()), -1)
        cv2.putText(
            im,
            bbox_text, (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 0, 0),
            1,
            lineType=cv2.LINE_AA)
    return Image.fromarray(im.astype('uint8'))
