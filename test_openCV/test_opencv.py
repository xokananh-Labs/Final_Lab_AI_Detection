import cv2
import os

def get_display_size(width: int, height: int, max_width: int = 1280, max_height: int = 720):
    """ຄຳນວນຂະໜາດສະແດງຜົນໃຫ້ພໍດີໜ້າຈໍ ໂດຍຮັກສາອັດຕາສ່ວນ"""
    if width <= max_width and height <= max_height:
        return width, height
    scale = min(max_width / width, max_height / height)
    return int(width * scale), int(height * scale)


def process_road_video(video_path: str, sample_interval_sec: float = 0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ບໍ່ສາມາດເປີດໄຟລ໌ວີດີໂອໄດ້")
        return

    # ສ້າງໂຟນເດີສໍາລັບການບັນທຶກ frame ທີ່ສຸ່ມ
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # ກໍລະນີກ້ອງບໍ່ສົ່ງຄ່າ FPS ຫຼື ຜິດປົກກະຕິ ໃຫ້ໃຊ້ 30.0 ເປັນຄ່າເລີ່ມຕົ້ນ
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0

    # ຈຳນວນ frame ຕໍ່ 1 ຄັ້ງທີ່ຕ້ອງການສຸ່ມ (ຕ່ຳສຸດແມ່ນ 1 frame)
    frame_step = max(1, round(fps * sample_interval_sec))

    disp_w, disp_h = get_display_size(width, height)
    is_fhd = (width >= 1920 and height >= 1080)

    print(f"Resolution: {width}x{height} {'(FHD+)' if is_fhd else ''}")
    print(f"FPS: {fps:.2f} | Duration: {duration_sec:.1f}s | Step: every {frame_step} frames (~{sample_interval_sec}s)")

    frame_idx = 0
    sampled_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            sampled_count += 1

            # ພະຍາຍາມດຶງເວລາຈິງຈາກວີດີໂອ (millisecond -> second); ຖ້າບໍ່ມີໃຫ້ຄິດຈາກ frame_idx / fps
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = (pos_msec / 1000.0) if pos_msec > 0 else (frame_idx / fps)

            # --- ເພີ່ມສ່ວນນີ້: ບັນທຶກເຟຣມເປັນ .jpg ລົງໂຟນເດີ output ---
            filename = f"frame_{sampled_count:05d}_t{timestamp_sec:06.2f}s.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            # --------------------------------------------------------

            # --- ຈຸດນີ້ແມ່ນບ່ອນທີ່ຈະສົ່ງ `frame` ໄປໃຫ້ AI ໃນອະນາຄົດ ---
            # ຕົວຢ່າງ: ຕັດເອົາສະເພາະສ່ວນລຸ່ມ 70% (ຕັດທ້ອງຟ້າອອກ) ແບບອີງຕາມ % ຄວາມສູງ
            roi_top = int(height * 0.30)
            roi_frame = frame[roi_top:height, 0:width]

            # ຂຽນຂໍ້ມູນສະຖານະລົງເທິງພາບເພື່ອກວດສອບ
            preview = cv2.resize(frame, (disp_w, disp_h))
            info_text = f"t={timestamp_sec:.2f}s | Frame={frame_idx} | Sampled={sampled_count}"
            cv2.putText(preview, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("FHD Multi-FPS Road Check", preview)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"ສຳເລັດ: ອ່ານທັງໝົດ {frame_idx} frames, ສຸ່ມມາປະມວນຜົນ {sampled_count} frames")


if __name__ == "__main__":
    # ປັບ sample_interval_sec ໄດ້ ເຊັ່ນ 0.5 (ທຸກ 0.5 ວິນາທີ) ຫຼື 1.0 (ທຸກ 1 ວິນາທີ)
    process_road_video("road.mp4", sample_interval_sec=0.5)