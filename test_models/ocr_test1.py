#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester (PT / ONNX / TFLite-LiteRT / TFLite-standard)

• Автоматично обробляє три типи TFLite-виходу:
  ① 4-тензорний   ② raw 41-атрибут   ③ post-sigmoid 7-атрибут.
• 1-D IoU-NMS вздовж X прибирає «стіну нулів».

Використання приклад:

    python3 ocr_test.py --model ocr.pt/onnx/tflite --image image.png --input_size 320 --conf 0.12 --iou_thr 0.35 --runs 5
    python3 ocr_test.py --model ocr.tflite --image image.png --tflite_backend standard

Залежності: opencv-python, numpy, (ai_edge_litert | tflite_runtime | tensorflow), ultralytics,
onnxruntime.  Python ≥ 3.8.
"""

from __future__ import annotations
import argparse, math, os, sys, time
from typing import List, Sequence, Tuple, Literal
import cv2
import numpy as np
from ultralytics import YOLO

# ─────────── parse_tflite_out ───────────────────────────────────────────────
def parse_tflite_out(interp, od: List[dict], thr: float, names: Sequence[str],
                     img_w: int, iou_thr: float,
                     tflite_backend: Literal['ai_edge', 'standard'] = 'ai_edge'):
    """Повертає [(x_center_px, char, conf), …] для LiteRT або стандартного TFLite."""
    def get_tensor_data(interpreter, tensor_details):
        if tflite_backend == 'ai_edge':
            return interpreter.get_tensor(tensor_details["index"])
        else: # standard tflite_runtime
            return interpreter.get_tensor(tensor_details['index'])

    def deq(a, d):
        if a.dtype == np.float32:
            return a
        # Для стандартного tflite_runtime, 'quantization_parameters' може бути порожнім,
        # якщо тензор не квантований, або містити списки.
        # 'quantization' містить (scale, zero_point)
        if 'quantization_parameters' in d and 'scales' in d['quantization_parameters']: # ai_edge_litert like
            qp = d["quantization_parameters"]
            scales = qp["scales"]
            zero_points = qp["zero_points"]
        elif 'quantization' in d and isinstance(d['quantization'], (list, tuple)) and len(d['quantization']) == 2 : # tflite_runtime like
            scales = np.array([d['quantization'][0]], dtype=np.float32)
            zero_points = np.array([d['quantization'][1]], dtype=np.int32) # або float32 залежно від версії
        else: # Немає параметрів квантування або невідомий формат
            return a.astype(np.float32)

        # Переконуємося, що scales та zero_points є скалярами або мають правильну форму
        scale = scales[0] if isinstance(scales, np.ndarray) and scales.ndim > 0 else scales
        zero_point = zero_points[0] if isinstance(zero_points, np.ndarray) and zero_points.ndim > 0 else zero_points

        return (a.astype(np.float32) - zero_point) * scale


    def iou1d(a1, a2):
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        return inter / ((a1[1]-a1[0]) + (a2[1]-a2[0]) - inter + 1e-6)

    det_raw: List[Tuple[float,float,str,float]] = []

    # 1) 4-тензорний
    if len(od) == 4 and od[0]["shape"][-1] == 4:
        boxes   = deq(get_tensor_data(interp, od[0]), od[0])[0]
        scores  = deq(get_tensor_data(interp, od[1]), od[1])[0]
        classes = deq(get_tensor_data(interp, od[2]), od[2])[0]
        n = int(get_tensor_data(interp, od[3])[0])
        for j in range(n):
            cf = float(scores[j]); cid = int(classes[j])
            if cf < thr or cid >= len(names):         continue
            x1, _, x2, _ = boxes[j]
            det_raw.append((x1, x2, names[cid], cf))

    # 2) один тензор
    else:
        raw = deq(get_tensor_data(interp, od[0]), od[0])
        if raw.ndim != 3:
            print(f"Помилка: вихідний тензор має несподівану розмірність {raw.ndim}. Очікується 3.", file=sys.stderr)
            return []
        attrs = raw.shape[2]

        # 2.a 6-атрибут
        if attrs == 6:
            for x1, _, x2, _, cf, cid_float in raw[0]: # cid може бути float
                cf = float(cf); cid = int(cid_float)
                if cf < thr or cid >= len(names):     continue
                det_raw.append((x1, x2, names[cid], cf))

        # 2.b 7- або 41-атрибут
        elif attrs >= 7:
            sigm = lambda x: 1/(1+math.exp(-x)) if x > -700 else 0 # запобігання overflow
            for row in raw[0]:
                if attrs == 7:  # post-sigmoid
                    cf  = float(row[4])
                    cid = int(row[5]) # cid може бути float
                    if cf < thr or cid >= len(names): continue
                    cx, w = float(row[0])*img_w, float(row[2])*img_w
                else:          # raw 41-attrib (або подібний)
                    obj = sigm(float(row[4]))
                    if obj < 1e-6:                   continue

                    # Перевірка, чи достатньо атрибутів для класів
                    num_classes = len(names)
                    if 5 + num_classes > attrs:
                        print(f"Помилка: недостатньо атрибутів ({attrs}) для {num_classes} класів.", file=sys.stderr)
                        continue # пропустити цей рядок

                    logits = row[5:5+num_classes]
                    cid = int(np.argmax(logits))
                    cf = obj * sigm(float(logits[cid]))
                    if cf < thr or cid >= len(names): continue
                    cx, w = float(row[0])*img_w, float(row[2])*img_w
                if w < 2:                           continue
                x1, x2 = cx - w/2, cx + w/2
                det_raw.append((x1, x2, names[cid], cf))
        else:
            print(f"Помилка: непідтримувана кількість атрибутів ({attrs}) у вихідному тензорі.", file=sys.stderr)
            return []


    if not det_raw:
        return []

    # NMS
    det_raw.sort(key=lambda d: d[3], reverse=True)
    keep: List[Tuple[float,float,str,float]] = []
    for cand in det_raw:
        is_kept = True
        for k_idx, k_val in enumerate(keep):
             # Перевірка, чи обмежуючі рамки суттєво відрізняються перед розрахунком IoU
            if abs(cand[0] - k_val[0]) > img_w and abs(cand[1] - k_val[1]) > img_w : # Якщо далеко, то IoU точно 0
                 continue
            if iou1d((cand[0],cand[1]), (k_val[0],k_val[1])) > iou_thr:
                # Якщо поточний кандидат має вищу впевненість, ніж збережений,
                # і вони перекриваються, можливо, варто замінити збережений.
                # Однак, стандартний NMS просто відкидає поточного кандидата.
                # Для простоти, дотримуємося стандартного підходу: відкидаємо поточного.
                is_kept = False
                break
        if is_kept:
            keep.append(cand)


    keep.sort(key=lambda d: (d[0]+d[1])/2)
    return [((k[0]+k[1])/2, k[2], k[3]) for k in keep]

# ─────────── решта допоміжних функцій ───────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess(img, inp_det: List[dict], img_sz: int,
               tflite_backend: Literal['ai_edge', 'standard'] = 'ai_edge'):
    # Для ai_edge_litert, inp_det[0] має 'shape', 'dtype', 'quantization_parameters'
    # Для tflite_runtime, inp_det[0] має 'shape', 'dtype', 'quantization' (scale, zero_point)
    details = inp_det[0]
    shp, dtype = details["shape"], details["dtype"]

    nchw = shp[1] == 3 if len(shp) == 4 else False # N H W C vs N C H W
    h_idx, w_idx = (2, 3) if nchw else (1, 2)

    h = shp[h_idx] if len(shp) == 4 and shp[h_idx] is not None else img_sz
    w = shp[w_idx] if len(shp) == 4 and shp[w_idx] is not None else img_sz

    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if len(img_resized.shape) == 2: # Якщо зображення монохромне
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4: # Якщо є альфа-канал
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
    elif img_resized.shape[2] == 3 and img.shape[2] == 3: # Перевірка, чи вже BGR
         # cv2.imread зазвичай завантажує як BGR
        pass # Вже BGR, не потрібно конвертувати в RGB тут, зробимо це пізніше, якщо треба
    else: # Якщо щось інше, пробуємо конвертувати з BGR в RGB
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)


    img_float = img_resized.astype(np.float32) / 255.0

    # Моделі YOLO часто очікують RGB
    if img.shape[2] == 3 and img_resized.shape[2] ==3 : # Якщо вхідне було кольоровим
         # Якщо img_resized ще не RGB, конвертуємо. cv2.imread дає BGR.
        # Однак, якщо модель була навчена на BGR, цей крок не потрібен.
        # Для узгодженості з більшістю YOLO моделей, припускаємо, що потрібен RGB.
        # Але якщо preprocess завантажує зображення як BGR, то YOLO.predict() обробить це.
        # Для TFLite, ми маємо бути точними.
        # Якщо inp_det вказує на RGB, то img_float має бути RGB.
        # Якщо на BGR, то img_float має бути BGR.
        # Зазвичай, TFLite моделі, конвертовані з PyTorch, очікують RGB.
        img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)


    if nchw:
        img_float = img_float.transpose(2, 0, 1) # HWC to CHW
    img_expanded = img_float[None] # Add batch dimension -> NCHW or NHWC

    if dtype in (np.int8, np.uint8):
        scale, zero_point = 1.0, 0 # Типові значення, якщо квантування не вказано

        if tflite_backend == 'ai_edge':
            if 'quantization_parameters' in details and details['quantization_parameters']['scales'] is not None:
                qp = details["quantization_parameters"]
                scale = qp["scales"][0] if qp["scales"].size > 0 else 1.0
                zero_point = qp["zero_points"][0] if qp["zero_points"].size > 0 else 0
        else: # standard tflite_runtime
            if 'quantization' in details and details['quantization'] != (0.0, 0):
                 # quantization is (scale, zero_point)
                scale = details['quantization'][0]
                zero_point = details['quantization'][1]

        if scale == 0: scale = 1.0 # Запобігання діленню на нуль

        img_quantized = (img_expanded / scale) + zero_point
        # Обрізка значень до діапазону dtype
        if dtype == np.int8:
            img_quantized = np.clip(img_quantized, -128, 127)
        elif dtype == np.uint8:
            img_quantized = np.clip(img_quantized, 0, 255)
        return img_quantized.astype(dtype)

    return img_expanded.astype(dtype)


def order_string(det):
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    # det вже має бути відсортований за x-координатою з parse_tflite_out
    # det.sort(key=lambda d: d[0]) # Не потрібно, якщо parse_tflite_out вже сортує
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(np.mean(confs)) if confs else 0.0

def load_names(pt_path):
    if pt_path:
        try:
            model_yolo = YOLO(pt_path)
            if model_yolo.names:
                # Переконуємося, що повертаємо список рядків
                return [str(name) for name in model_yolo.names.values()] \
                       if isinstance(model_yolo.names, dict) else \
                       [str(name) for name in model_yolo.names]
        except Exception as e:
            print(f"Не вдалося завантажити імена класів з {pt_path}: {e}", file=sys.stderr)
            pass
    return CLASS_NAMES

# ─────────── back-ends ──────────────────────────────────────────────────────
def run_pt_or_onnx(path, img, names, thr, runs):
    m = YOLO(path); _ = m(img, verbose=False, conf=thr) # warm-up
    times, det_accum = [], []
    final_det_for_ordering = []

    for i in range(runs):
        t0 = time.perf_counter()
        r = m(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter()-t0)
        current_run_det = []
        if r.boxes:
            for b in r.boxes:
                cf=float(b.conf.squeeze()); cid=int(b.cls.squeeze())
                if cf<thr or cid>=len(names): continue
                # b.xywh містить [x_center, y_center, width, height]
                # Нам потрібен x_center для сортування
                current_run_det.append((float(b.xywh.squeeze()[0]), names[cid], cf))
        if i == runs-1: # Беремо детекції з останнього прогону для результату
            final_det_for_ordering = current_run_det

    return *order_string(final_det_for_ordering), np.mean(times)


def run_litert(path, img, names, thr, runs, img_sz, iou_thr,
               tflite_backend: Literal['ai_edge', 'standard']):
    if tflite_backend == 'ai_edge':
        from ai_edge_litert.interpreter import Interpreter as AiEdgeInterpreter
        # Примітка: XNNPACK зазвичай вмикається за замовчуванням, якщо доступний.
        # Для ai_edge_litert це може залежати від збірки та платформи.
        it = AiEdgeInterpreter(model_path=path)
    else: # 'standard'
        try:
            from tflite_runtime.interpreter import Interpreter as StdInterpreter
            from tflite_runtime.interpreter import load_delegate
        except ImportError:
            print("Помилка: tflite_runtime не встановлено. Встановіть 'pip install tflite-runtime' або використовуйте --tflite_backend ai_edge.", file=sys.stderr)
            return "ПОМИЛКА RUNTIME", 0.0, 0.0

        # Спроба завантажити делегат XNNPACK, якщо доступний, для стандартного runtime
        delegates = []
        try:
            # На деяких платформах XNNPACK може бути вже вбудований або не вимагати явного завантаження.
            # На Raspberry Pi (де часто використовується tflite_runtime), XNNPACK зазвичай використовується
            # автоматично, якщо TFLite був зібраний з його підтримкою.
            # Явне завантаження делегата може бути потрібне для інших платформ або для контролю параметрів.
            # xnnpack_delegate = load_delegate('libtensorflowlite_flex_delegate.so') # Приклад для Flex delegate
            # Для XNNPACK, якщо не вбудований, шлях до .so може бути потрібен.
            # Однак, найчастіше XNNPACK вже є частиною runtime.
            # Якщо ви хочете спробувати вимкнути XNNPACK, це робиться через Interpreter.Options (C++ API)
            # або може не бути доступно напряму з Python API tflite_runtime без спеціальної збірки.
            # Ми просто створимо Interpreter, він має використовувати XNNPACK, якщо може.
            pass
        except Exception as e:
            print(f"Попередження: не вдалося завантажити/налаштувати делегат: {e}", file=sys.stderr)

        it = StdInterpreter(model_path=path, experimental_delegates=delegates if delegates else None)

    it.allocate_tensors()
    idet, odet = it.get_input_details(), it.get_output_details()

    # Важливо: переконайтеся, що preprocess відповідає тому, як викликається get_input_details
    inp = preprocess(img, idet, img_sz, tflite_backend=tflite_backend)

    # Прогрів
    it.set_tensor(idet[0]["index"], inp)
    it.invoke()

    times, det = [], []
    for i in range(runs):
        t0=time.perf_counter()
        it.set_tensor(idet[0]["index"], inp)
        it.invoke()
        times.append(time.perf_counter()-t0)
        if i==runs-1: # Беремо детекції з останнього прогону
            # Передаємо tflite_backend в parse_tflite_out
            det=parse_tflite_out(it, odet, thr, names, img_w=img_sz,
                                 iou_thr=iou_thr, tflite_backend=tflite_backend)
    return *order_string(det), np.mean(times)

# ─────────── CLI та main ────────────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser(description="OCR-YOLO tester (LiteRT / Standard TFLite)",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-m","--model","--path",dest="model_paths",
                   action="append",required=True, help="Шлях до файлу моделі (.pt, .onnx, .tflite)")
    p.add_argument("--image",required=True, help="Шлях до зображення для тестування")
    p.add_argument("--input_size",type=int,default=320, help="Розмір входу для моделі (ширина)")
    p.add_argument("--conf",type=float,default=0.12, help="Поріг впевненості для детекцій")
    p.add_argument("--iou_thr",type=float,default=0.35, help="Поріг IoU для Non-Maximum Suppression (NMS)")
    p.add_argument("--runs",type=int,default=5, help="Кількість тестових запусків для вимірювання часу")
    p.add_argument("--tflite_backend", type=str, default="ai_edge", choices=["ai_edge", "standard"],
                   help="Тип TFLite runtime для використання з .tflite моделями.")
    return p.parse_args()

if __name__=="__main__":
    args=cli()
    for f in args.model_paths + ([args.image] if args.image else []):
        if not os.path.exists(f):
            sys.exit(f"Файл не знайдено: {f}")

    img_orig = cv2.imread(args.image)
    if img_orig is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    # Завантажуємо імена класів. Спробуємо з першої .pt моделі, якщо є.
    pt_first = next((p for p in args.model_paths if p.lower().endswith(".pt")), None)
    names = load_names(pt_first)
    if not names or names == CLASS_NAMES: # Якщо з .pt не вдалося або повернуло стандартні
        print("Використовуються стандартні імена класів (0-9, A-Z).", file=sys.stderr)


    print(f"\n{f'[ Тестування з {args.runs} запусками ]':^38}")
    print("="*38)
    for mp in args.model_paths:
        # Робимо копію зображення для кожної моделі, оскільки деякі pre/post processing
        # можуть теоретично модифікувати його (хоча в цьому скрипті не повинні)
        img_copy = img_orig.copy()
        model_name = os.path.basename(mp)
        ext = os.path.splitext(mp)[1].lower()
        plate, conf, t = "ПОМИЛКА", 0.0, 0.0
        valid_run = False

        try:
            if ext in(".pt",".onnx"):
                # Переконуємося, що names завантажені, особливо якщо немає .pt файлу
                if names == CLASS_NAMES and not pt_first: # якщо pt_first не було, load_names повернув CLASS_NAMES
                    # Для .pt або .onnx, якщо є метадані, YOLO() їх завантажить.
                    # Якщо ми тут, імена або стандартні, або не завантажились.
                    # YOLO() сама впорається із завантаженням імен з моделі, якщо вони там є.
                    # Якщо імен немає в моделі, вона може використати свої стандартні (напр. COCO).
                    # Тому передача names тут важлива, якщо ми хочемо наші CLASS_NAMES.
                     pass # names вже встановлено

                plate,conf,t = run_pt_or_onnx(mp,img_copy,names,args.conf,args.runs)
                valid_run = True
            elif ext==".tflite":
                plate,conf,t = run_litert(mp,img_copy,names,args.conf,args.runs,
                                        img_sz=args.input_size,
                                        iou_thr=args.iou_thr,
                                        tflite_backend=args.tflite_backend)
                valid_run = True
            else:
                print(f"[{model_name:>16}]  ❌ Формат не підтримується.")
                continue

            if valid_run:
                 print(f"[{model_name:>16}]  Plate: {plate:<18} "
                       f"Avg conf: {conf:.3f}  Time: {t*1000:.1f} ms")

        except Exception as e:
            print(f"Помилка при обробці {model_name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


    print("="*38 + "\n")