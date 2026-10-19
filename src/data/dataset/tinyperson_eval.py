"""
HiVa-DETR

The TinyPerson AP protocol of the TinyBenchmark repository (Yu et al., WACV 2020), ported from
its ``tiny_benchmark/maskrcnn_benchmark/data/datasets/evaluation/coco/cocoeval.py`` as
``evaluate_tiny.py`` runs it (``eval_standard='tiny'``, ``ignore_uncertain=True``,
``use_iod_for_ignore=True``). It differs from COCOeval in four ways, all kept here:

- IoU thresholds 0.25, 0.5 and 0.75, and the size ranges all [1, 1e5], tiny [1, 20], tiny1
  [1, 8], tiny2 [8, 12], tiny3 [12, 20], small [20, 32] and reasonable [32, 1e5], as the square
  root of the area in pixels.
- Up to 200 detections per image in TinyBenchmark. Its successor TOV_mmdetection, the codebase
  the authors now recommend, runs the same matching (``huicv``'s ``COCOExpandEval``, identical
  with the TinyPerson configs' ``iod_th_of_iou_f`` of ``lambda iou: iou``) with 1000
  (``evaluation.proposal_nums``), which its own table puts at +1.5 AP50 on tiny for one FCOS
  model. ``max_dets`` takes either or both; the summary is at the largest.
- Ignore regions are boxes with ``ignore`` set (TinyPerson's ignore and uncertain boxes), not
  crowd boxes: their IoU is the plain IoU, and each can absorb one detection by it.
- A detection that matches nothing by IoU is still ignored when it lies inside an ignored box,
  by intersection over the detection's own area at the same threshold as the IoU. The ignored
  boxes here include the objects outside the size range being scored, as the original has it.
- The summary is AP and AR at each threshold and size range (42 numbers).

The matching is the original's greedy loop, decision for decision, with the candidates of each
detection found with numpy first; on the official FreeAnchor submission it reproduces the
original's 42 numbers and its precision and recall arrays exactly.
"""

import copy
import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

from ...core import register
from .coco_eval import CocoEvaluator

__all__ = ["TinyPersonCOCOeval", "TinyPersonEvaluator"]

IOU_THRESHOLDS = (0.25, 0.5, 0.75)
AREA_RANGES = (
    ("all", 1**2, 1e5**2),
    ("tiny", 1**2, 20**2),
    ("tiny1", 1**2, 8**2),
    ("tiny2", 8**2, 12**2),
    ("tiny3", 12**2, 20**2),
    ("small", 20**2, 32**2),
    ("reasonable", 32**2, 1e5**2),
)
MAX_DETS = (200,)  # TinyBenchmark's; TOV_mmdetection's TinyPerson configs use 1000

# the headline numbers at the largest detection limit, in the order of ``stats`` (entry 0, AP50 on
# tiny, first), followed by AP50 on tiny at every smaller
# limit: (ap, iou threshold, size range)
HEADLINE = (
    (1, 0.5, "tiny"),
    (1, 0.5, "tiny1"),
    (1, 0.5, "tiny2"),
    (1, 0.5, "tiny3"),
    (1, 0.5, "small"),
    (1, 0.25, "tiny"),
    (1, 0.75, "tiny"),
    (1, 0.5, "all"),
    (0, 0.5, "tiny"),
)


def headline_labels(max_dets):
    """Names of the ``stats`` entries for these detection limits: the headline at the largest, then AP50 on tiny at the rest."""
    *smaller, largest = sorted(max_dets)
    labels = [f"{'AP' if ap else 'AR'}{int(t * 100)}_{area}" for ap, t, area in HEADLINE]
    return labels + [f"AP50_tiny@{m}" for m in smaller]


def _iou_xywh(dt, gt):
    """``[D, G]`` IoU of xywh boxes, as pycocotools' bbox IoU with no crowd box."""
    dx1, dy1, dw, dh = dt[:, 0:1], dt[:, 1:2], dt[:, 2:3], dt[:, 3:4]
    gx1, gy1, gw, gh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    iw = np.minimum(dx1 + dw, gx1 + gw) - np.maximum(dx1, gx1)
    ih = np.minimum(dy1 + dh, gy1 + gh) - np.maximum(dy1, gy1)
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    union = dw * dh + gw * gh - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


class TinyPersonCOCOeval:
    """
    The TinyBenchmark evaluation over a COCO ground truth whose annotations carry ``bbox`` (xywh),
    ``area``, ``category_id``, ``id`` and ``ignore`` (``iscrowd`` is read as ignore when there is
    no ``ignore``). ``evaluate`` takes the detections of every image, ``accumulate`` and
    ``summarize`` produce ``eval`` and ``stats`` as COCOeval does.
    """

    def __init__(self, coco_gt, iou_type="bbox", print_function=print, max_dets=MAX_DETS):
        if iou_type != "bbox":
            raise ValueError(f"TinyPerson evaluation is bbox only, got iou_type={iou_type}")
        self.cocoGt = coco_gt
        self.print_function = print_function
        self.params = SimpleNamespace(
            iouType=iou_type,
            imgIds=sorted(coco_gt.getImgIds()),
            catIds=sorted(coco_gt.getCatIds()),
            iouThrs=np.array(IOU_THRESHOLDS),
            recThrs=np.linspace(0.0, 1.00, 101, endpoint=True),
            maxDets=sorted(int(m) for m in max_dets),
            areaRng=[[lo, hi] for _, lo, hi in AREA_RANGES],
            areaRngLbl=[name for name, _, _ in AREA_RANGES],
            useCats=1,
        )
        self.evalImgs = []
        self.eval = {}
        self.stats = np.zeros((len(headline_labels(self.params.maxDets)),))
        self.all_stats = np.zeros((2 * len(IOU_THRESHOLDS) * len(AREA_RANGES),))

    def _ground_truth(self, image_id, category_id):
        """One image's boxes of one category: xywh, area, ignore flag and id, in annotation order."""
        anns = [a for a in self.cocoGt.imgToAnns.get(image_id, []) if a["category_id"] == category_id]
        boxes = np.array([a["bbox"] for a in anns], dtype=np.float64).reshape(-1, 4)
        area = np.array([a["area"] for a in anns], dtype=np.float64)
        ignore = np.array([bool(a.get("ignore", a.get("iscrowd", 0))) for a in anns], dtype=bool)
        return boxes, area, ignore

    def evaluate(self, detections):
        """
        Match every image's detections. ``detections`` maps an image id to ``(boxes, scores,
        labels)``: boxes ``[N, 4]`` xywh in pixels, scores ``[N]``, labels ``[N]`` category ids.
        Images of the ground truth that have no entry have no detections.
        """
        p = self.params
        max_det = p.maxDets[-1]
        self.evalImgs = []
        for category_id in p.catIds:
            per_image = []
            for image_id in p.imgIds:
                gt_boxes, gt_area, gt_ignore = self._ground_truth(image_id, category_id)
                boxes, scores, labels = detections.get(image_id, (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))))
                keep = np.asarray(labels) == category_id
                dt_boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[keep]
                dt_scores = np.asarray(scores, dtype=np.float64)[keep]
                # highest score first, stable, cut to the detection limit (computeIoU and evaluateImg)
                order = np.argsort(-dt_scores, kind="mergesort")[:max_det]
                dt_boxes, dt_scores = dt_boxes[order], dt_scores[order]
                ious = _iou_xywh(dt_boxes, gt_boxes) if len(dt_boxes) and len(gt_boxes) else None
                per_image.append((gt_boxes, gt_area, gt_ignore, dt_boxes, dt_scores, ious))
            for _, lo, hi in AREA_RANGES:
                for entry in per_image:
                    self.evalImgs.append(self._evaluate_image(*entry, lo, hi))
        self._paramsEval = copy.deepcopy(p)

    def _evaluate_image(self, gt_boxes, gt_area, gt_ignore, dt_boxes, dt_scores, ious, lo, hi):
        """The original ``evaluateImg`` for one image, category and size range."""
        G, D = len(gt_boxes), len(dt_boxes)
        if G == 0 and D == 0:
            return None
        T = len(IOU_THRESHOLDS)

        # objects outside the size range are ignored like the ignore boxes, and sorted after the rest
        gt_ig_all = gt_ignore | (gt_area < lo) | (gt_area > hi)
        gt_order = np.argsort(gt_ig_all, kind="mergesort")
        gt_ig = gt_ig_all[gt_order]

        dtm = np.zeros((T, D), dtype=bool)  # matched (to a regular or an ignored box)
        dt_ig = np.zeros((T, D), dtype=bool)
        if ious is not None:
            ious = ious[:, gt_order]
            ignore_idx = np.nonzero(gt_ig)[0]
            if len(ignore_idx):
                # intersection over the detection's area, from the IoU, as IOD_by_IOU computes it
                ignore_area = gt_area[gt_order][ignore_idx][None, :]
                dt_area = (dt_boxes[:, 2] * dt_boxes[:, 3])[:, None]
                ignore_iou = ious[:, ignore_idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    iods = ignore_iou / (1 + ignore_iou) * (1 + ignore_area / dt_area)
                iod_best = np.argmax(iods, axis=1)
                iod_max = iods[np.arange(D), iod_best]
            for t, thr in enumerate(IOU_THRESHOLDS):
                thr = min(thr, 1 - 1e-10)
                gtm = np.zeros((G,), dtype=bool)
                rows, cols = np.nonzero(ious >= thr)
                starts = np.searchsorted(rows, np.arange(D + 1))
                for d in range(D):
                    m, best = -1, thr
                    # the candidates in ground-truth order: regular boxes first, then ignored ones
                    for g in cols[starts[d] : starts[d + 1]].tolist():
                        if gtm[g]:
                            continue
                        if m > -1 and not gt_ig[m] and gt_ig[g]:
                            break
                        iou = ious[d, g]
                        if iou < best:
                            continue
                        best, m = iou, g
                    if m == -1:
                        if len(ignore_idx) == 0 or not iod_max[d] >= thr:
                            continue
                        m = ignore_idx[iod_best[d]]
                    dt_ig[t, d] = gt_ig[m]
                    dtm[t, d] = True
                    gtm[m] = True

        # unmatched detections outside the size range are ignored
        dt_area = dt_boxes[:, 2] * dt_boxes[:, 3]
        out_of_range = (dt_area < lo) | (dt_area > hi)
        dt_ig = dt_ig | (~dtm & out_of_range[None, :])
        return {"dtScores": dt_scores, "dtMatches": dtm, "dtIgnore": dt_ig, "gtIgnore": gt_ig}

    def accumulate(self):
        """The original ``accumulate``: precision at 101 recall points and recall, per threshold and size range."""
        p = self.params
        T, R, K, A, M = len(p.iouThrs), len(p.recThrs), len(p.catIds), len(p.areaRng), len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))
        I0 = len(p.imgIds)
        for k in range(K):
            for a in range(A):
                for m, max_det in enumerate(p.maxDets):
                    E = [e for e in self.evalImgs[(k * A + a) * I0 : (k * A + a + 1) * I0] if e is not None]
                    if not E:
                        continue
                    dt_scores = np.concatenate([e["dtScores"][0:max_det] for e in E])
                    inds = np.argsort(-dt_scores, kind="mergesort")
                    dt_scores_sorted = dt_scores[inds]
                    dtm = np.concatenate([e["dtMatches"][:, 0:max_det] for e in E], axis=1)[:, inds]
                    dt_ig = np.concatenate([e["dtIgnore"][:, 0:max_det] for e in E], axis=1)[:, inds]
                    gt_ig = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gt_ig == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dt_ig))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ig))
                    tp_sum = np.cumsum(tps, axis=1).astype(float)
                    fp_sum = np.cumsum(fps, axis=1).astype(float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        recall[t, k, a, m] = rc[-1] if nd else 0
                        pr = np.maximum.accumulate(pr[::-1])[::-1] if nd else pr
                        # precision at each recall threshold, 0 past the last recall reached
                        idx = np.searchsorted(rc, p.recThrs, side="left")
                        valid = idx < nd
                        q = np.zeros((R,))
                        ss = np.zeros((R,))
                        q[valid] = pr[idx[valid]]
                        ss[valid] = dt_scores_sorted[idx[valid]]
                        precision[t, :, k, a, m] = q
                        scores[t, :, k, a, m] = ss
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }

    def _summarize(self, ap, iou_threshold, area, m=-1):
        p = self.params
        t = np.where(np.abs(iou_threshold - p.iouThrs) < 1e-6)[0]
        a = p.areaRngLbl.index(area)
        s = self.eval["precision"][t][:, :, :, a, m] if ap else self.eval["recall"][t][:, :, a, m]
        return float(np.mean(s[s > -1])) if (s > -1).any() else -1.0

    def summarize(self):
        """The original's 42 numbers at the largest detection limit (AP, then AR, by threshold, by size range), and the headline."""
        if not self.eval:
            raise Exception("Please run accumulate() first")
        p = self.params
        rows = []
        for ap in (1, 0):
            for iou_threshold in IOU_THRESHOLDS:
                for area in p.areaRngLbl:
                    value = self._summarize(ap, iou_threshold, area)
                    rows.append(value)
                    title = "Average Precision  (AP)" if ap else "Average Recall     (AR)"
                    self.print_function(
                        f" {title} @[ IoU={iou_threshold:<9.2f} | area={area:>10s} | maxDets={p.maxDets[-1]:>3d} ] = {value:0.4f}"
                    )
        self.all_stats = np.array(rows)
        stats = [self._summarize(ap, iou_threshold, area) for ap, iou_threshold, area in HEADLINE]
        stats += [self._summarize(1, 0.5, "tiny", m) for m in range(len(p.maxDets) - 1)]
        self.stats = np.array(stats)


@register()
class TinyPersonEvaluator(CocoEvaluator):
    """
    The TinyPerson protocol: the detections of an image scored by ``TinyPersonCOCOeval``, after
    the windows of a tiled validation split (``TinyPersonDetection`` with ``tile_size``) are put
    back together, the way ``evaluate_tiny.py`` merges sub-image results: every window's
    detections shifted into the image and clipped to the window, then one class-wise NMS over the
    image at ``merge_nms_threshold`` (the original's 0.5).

    The windows are read from the ground truth (``coco_gt.dataset['tiles']``); without them the
    keys of ``update`` are image ids and nothing is merged. Detections are kept per key until
    ``synchronize_between_processes``, since an image is complete only once all its windows
    arrived. The images scored are those with at
    least one window seen. ``max_dets`` are the detection limits per image (see the module
    docstring); ``stats`` is the headline at the largest (AP50 on tiny first, see
    ``headline_labels``) and ``all_stats`` the original's 42 numbers there.
    """

    def __init__(self, coco_gt, iou_types=("bbox",), merge_nms_threshold=0.5, max_dets=MAX_DETS):
        if list(iou_types) != ["bbox"]:
            raise ValueError(f"TinyPerson evaluation is bbox only, got {iou_types}")
        self.merge_nms_threshold = merge_nms_threshold
        self.max_dets = list(max_dets)
        super().__init__(coco_gt, list(iou_types))
        tiles = self.coco_gt.dataset.get("tiles")
        self.tiles = None if tiles is None else np.asarray(tiles, dtype=np.int64)

    def _build_coco_eval(self, iou_type):
        return TinyPersonCOCOeval(self.coco_gt, iou_type=iou_type, print_function=print, max_dets=self.max_dets)

    def cleanup(self):
        super().cleanup()
        self.predictions = {}

    def update(self, predictions):
        for key, prediction in predictions.items():
            boxes = prediction["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
            scores = prediction["scores"].detach().float().cpu().numpy().reshape(-1)
            labels = prediction["labels"].detach().cpu().numpy().reshape(-1)
            if self.tiles is not None:
                _, x, y, w, h = self.tiles[key]
                boxes = boxes + np.array([x, y, x, y], dtype=boxes.dtype)
                boxes[:, 0::2] = boxes[:, 0::2].clip(x, x + w)
                boxes[:, 1::2] = boxes[:, 1::2].clip(y, y + h)
            self.predictions[int(key)] = (boxes, scores, labels)

    def synchronize_between_processes(self):
        merged = self.predictions

        by_image = {}
        for key in sorted(merged):
            image_id = int(self.tiles[key][0]) if self.tiles is not None else key
            by_image.setdefault(image_id, []).append(merged[key])
        detections = {}
        for image_id, parts in by_image.items():
            boxes = np.concatenate([p[0] for p in parts])
            scores = np.concatenate([p[1] for p in parts])
            labels = np.concatenate([p[2] for p in parts])
            if self.tiles is not None and len(parts) > 1 and len(boxes):
                keep = torchvision.ops.batched_nms(
                    torch.from_numpy(boxes),
                    torch.from_numpy(scores),
                    torch.from_numpy(labels),
                    self.merge_nms_threshold,
                ).numpy()
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
            detections[image_id] = (xywh, scores, labels)

        for coco_eval in self.coco_eval.values():
            coco_eval.params.imgIds = sorted(detections)
            coco_eval.evaluate(detections)
        self.img_ids = sorted(detections)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
            labels = headline_labels(coco_eval.params.maxDets)
            headline = ", ".join(f"{name} {v * 100:.2f}" for name, v in zip(labels, coco_eval.stats))
            print(f"TinyPerson (maxDets {coco_eval.params.maxDets[-1]}): {headline}")
