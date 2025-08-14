async def _gen_file_mjpeg_backup(sess_id: str, rate: float):
    """
    Backup of original heavy _gen_file_mjpeg implementation from api/app.py.
    Kept as a restoration point — do not import automatically.
    """
    import time as _time
    global FILE_MODEL, FILE_MODEL_PATH, MODEL_PATH
    frame_idx_local = 0
    while True:
        try:
            sess = _file_sessions.get(sess_id)
            if not sess or sess.get("type") != "local":
                break

            cap = sess.get("cap")
            if cap is None:
                _time.sleep(0.02)
                continue

            fps = max(5.0, float(sess.get("fps", 25.0) or 25.0))
            t0 = _time.time()
            # измерим стадии: seek (для play это read), infer, encode, total
            t_seek0 = _time.time()
            frame = _read_frame_local(cap)
            seek_ms = (time.time() - t_seek0) * 1000.0
            if frame is None:
                _time.sleep(max(0.005, 1.0 / fps))
                continue

            do_infer = (frame_idx_local % max(1, FRAME_SKIP) == 0)
            det_count = int(sess.get("last_count", 0) or 0)
            if do_infer:
                # ensure model
                from ultralytics import YOLO
                try:
                    target_model_path = str((MODELS_DIR / MODEL_PATH.split("/")[-1]))
                except Exception:
                    target_model_path = os.getenv("MODEL_PATH", "models/yolo11n-seg.pt")
                if (FILE_MODEL is None) or (FILE_MODEL_PATH != target_model_path):
                    logger.info("[file_play] Loading YOLO model: %s", target_model_path)
                    FILE_MODEL = YOLO(target_model_path)
                    FILE_MODEL_PATH = target_model_path
                # predict
                t_inf0 = time.time()
                results = FILE_MODEL.predict(
                    frame,
                    imgsz=640,
                    conf=CONF_THRESHOLD,
                    verbose=False,
                    retina_masks=True
                )
                infer_ms = (time.time() - t_inf0) * 1000.0
                if results and len(results) > 0:
                    r = results[0]
                    det_bboxes = []
                    det_idx_map = []
                    if hasattr(r, "boxes") and r.boxes is not None:
                        xyxy = r.boxes.xyxy
                        cls = r.boxes.cls
                        conf = r.boxes.conf
                        if hasattr(xyxy, "cpu"):
                            xyxy = xyxy.cpu().numpy()
                        if hasattr(cls, "cpu"):
                            cls = cls.cpu().numpy()
                        if hasattr(conf, "cpu"):
                            conf = conf.cpu().numpy()
                        for i, b in enumerate(xyxy):
                            c = int(cls[i]) if i < len(cls) else -1
                            cf = float(conf[i]) if i < len(conf) else 0.0
                            if (c in TARGET_CLASS_IDS) and (cf >= CONF_THRESHOLD):
                                x1, y1, x2, y2 = map(float, b)
                                det_bboxes.append([x1, y1, x2, y2])
                                det_idx_map.append(i)
                tracker = sess.get("tracker")
                tracks = tracker.update([det_bboxes[k] + [det_idx_map[k]] for k in range(len(det_bboxes))]) if tracker else []
                det_count = len(tracks)
                # draw masks
                mask_data = None
                if hasattr(r, "masks") and r.masks is not None:
                    mask_data = r.masks.data
                    if hasattr(mask_data, "cpu"):
                        mask_data = mask_data.cpu().numpy()
                    h, w = frame.shape[:2]
                    overlay = np.zeros_like(frame)
                    for tr in tracks:
                        tid = tr['id']
                        mi = tr.get('det_index', -1)
                        if 0 <= mi < len(mask_data):
                            mask = (mask_data[mi] > 0.5).astype(np.uint8)
                            if mask.shape[:2] != (h, w):
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            color = _pastel_color_for_id(int(tid))
                            overlay[mask > 0] = color
                    sess["overlay_image"] = overlay
                    sess["overlay_shape"] = (h, w)
                    # optional ID labels kept minimal to reduce cost
                    try:
                        ys, xs = np.where(mask > 0)
                        if len(xs) > 0:
                            cx, cy = int(xs.mean()), int(ys.mean())
                            cv2.putText(frame, str(int(tid)), (max(0, cx-10), max(12, cy-8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                # cache
                sess["_last_tracks"] = tracks if do_infer else sess.get("_last_tracks")
                sess["_last_masks"] = getattr(r, 'masks', None) if do_infer else sess.get("_last_masks")
                # update rolling avg
                sess["last_count"] = int(det_count)
                cw = sess.setdefault("count_window", [])
                cw.append(float(det_count))
                if len(cw) > AVG_WINDOW:
                    cw.pop(0)
                sess["avg_count"] = (sum(cw) / len(cw)) if cw else 0.0
            else:
                # overlay last
                mask_data = sess.get("_last_masks")
                if mask_data is not None:
                    try:
                        if hasattr(mask_data, 'cpu'):
                            mask_data = mask_data.cpu().numpy()
                        tracks = sess.get("_last_tracks") or []
                        h, w = frame.shape[:2]
                        overlay = np.zeros_like(frame)
                        for tr in tracks:
                            tid = tr['id']
                            mi = tr.get('det_index', -1)
                            if 0 <= mi < len(mask_data):
                                mask = (mask_data[mi] > 0.5).astype(np.uint8)
                                if mask.shape[:2] != (h, w):
                                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                color = _pastel_color_for_id(int(tid))
                                overlay[mask > 0] = color
                        sess["overlay_image"] = overlay
                        sess["overlay_shape"] = (h, w)
                    except Exception:
                        pass

            # pacing and encoding
            spent = time.time() - t0
            target = max(0.005, (1.0 / fps))
            await asyncio.sleep(max(0.0, target - spent))
            frame_idx_local += 1
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("[file_play backup] error")
    finally:
        # Закрываем ресурсы при выходе из функции
        pass