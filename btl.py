import cv2
import sqlite3
import os
import re
from datetime import datetime
from ultralytics import YOLO
import easyocr


# 1. CẤU HÌNH HỆ THỐNG

DB_NAME = "parking_system.db"
IMAGE_DIR = "saved_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

print("[INFO] Đang tải mô hình AI...")
try:
    plate_detector = YOLO('best.pt')
except:
    plate_detector = YOLO('yolov8n.pt')
ocr_reader = easyocr.Reader(['en'], gpu=True)
print("[INFO] Đã tải xong AI!")


# 2. XỬ LÝ CƠ SỞ DỮ LIỆU

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS PARKING_SESSIONS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        license_plate_in TEXT, 
                        image_in TEXT, 
                        check_in_time DATETIME, 
                        status TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS LOGS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        session_id INTEGER, 
                        license_plate_out TEXT, 
                        image_out TEXT, 
                        check_out_time DATETIME, 
                        confidence REAL, 
                        match_result TEXT)''')
    conn.commit()
    conn.close()

def execute_query(query, params=()):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def fetch_one(query, params=()):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchone()
    conn.close()
    return result

def fetch_all(query, params=()):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.close()
    return result


# 3. HÀM NHẬN DIỆN & XỬ LÝ BIỂN SỐ

def normalize_plate(plate_str):
    return re.sub(r'[^A-Z0-9]', '', plate_str.upper())

def read_plate_from_frame(frame):
    # 1. AI YOLO cố gắng khoanh vùng vị trí biển số
    results = plate_detector(frame, verbose=False)
    best_plate_text = ""
    best_conf = 0.0
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0: continue
                
                # Tiền xử lý ảnh nâng cao để khử nhiễu màn hình
                plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Dùng Bilateral Filter để làm mờ nhiễu sọc nhưng vẫn giữ sắc nét viền chữ
                bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
                # Tăng độ tương phản bằng Otsu Threshold
                _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                ocr_result = ocr_reader.readtext(thresh, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=True)
                raw_text = "".join(ocr_result)
                
                best_plate_text = normalize_plate(raw_text)
                best_conf = conf

    # 2. PHƯƠNG ÁN DỰ PHÒNG (Fallback) khi dí sát camera
    if not best_plate_text:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter_full = cv2.bilateralFilter(gray_full, 11, 17, 17)
        _, thresh_full = cv2.threshold(bfilter_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        ocr_result_fallback = ocr_reader.readtext(thresh_full, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=True)
        raw_text_fb = "".join(ocr_result_fallback)
        
        normalized_text = normalize_plate(raw_text_fb)
        if len(normalized_text) >= 5: 
            best_plate_text = normalized_text
            best_conf = 0.85

    return best_plate_text, best_conf
def save_image(frame, prefix, plate):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{plate}_{timestamp}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath


# 4. LOGIC VÀO / RA BÃI

def process_check_in(frame):
    plate_str, conf = read_plate_from_frame(frame)
    if not plate_str: return "Loi: Khong nhan dien duoc bien so", False
    if conf < 0.3: return f"Loi: Nhan dien kem ({plate_str})", False

    session = fetch_one("SELECT id FROM PARKING_SESSIONS WHERE license_plate_in=? AND status='Đang gửi'", (plate_str,))
    if session:
        return f"Tu choi: Xe [{plate_str}] dang o trong bai!", False

    img_path = save_image(frame, "IN", plate_str)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    execute_query("INSERT INTO PARKING_SESSIONS (license_plate_in, image_in, check_in_time, status) VALUES (?, ?, ?, ?)", 
                  (plate_str, img_path, now, "Đang gửi"))
    
    return f"VAO THANH CONG: [{plate_str}]", True

def process_check_out(frame):
    plate_str, conf = read_plate_from_frame(frame)
    if not plate_str: return "Loi: Khong nhan dien duoc bien so", False

    session = fetch_one("SELECT id FROM PARKING_SESSIONS WHERE license_plate_in=? AND status='Đang gửi'", (plate_str,))
    if not session:
         return f"Loi: Xe [{plate_str}] khong co trong bai!", False

    img_path = save_image(frame, "OUT", plate_str)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = session[0]
    
    execute_query("UPDATE PARKING_SESSIONS SET status='Đã ra' WHERE id=?", (session_id,))
    execute_query("INSERT INTO LOGS (session_id, license_plate_out, image_out, check_out_time, confidence, match_result) VALUES (?, ?, ?, ?, ?, ?)", 
                  (session_id, plate_str, img_path, now, conf, "Khớp"))
    
    return f"RA THANH CONG: [{plate_str}]", True


# 5. GIAO DIỆN CAMERA CHÍNH (MAIN LOOP)

def main():
    init_db() 
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("[LỖI] Không thể mở camera!")
        return

    status_message = "He thong san sang. Cho lenh..."
    status_color = (0, 255, 0) 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # BẢN SAO SẠCH: Dùng để gửi cho AI đọc (chưa bị vẽ chữ lên)
        clean_frame = frame.copy()

        #Vẽ UI lên màn hình Camera 
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 130), (0, 0, 0), -1) 
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "HUNG DAN SU DUNG:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "Phim [V]: Chup hinh xe VAO", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Phim [R]: Chup hinh xe RA", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Phim [L]: Xem danh sach xe", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "Phim [Q]: Thoat chuong trinh", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Trang thai: {status_message}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.imshow("He Thong Quan Ly Bai Xe - AI ALPR", frame)
        

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('v'):
            status_message = "Dang doc bien so xe VAO..."
            status_color = (0, 255, 255)
            cv2.putText(frame, status_message, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow("He Thong Quan Ly Bai Xe - AI ALPR", frame)
            cv2.waitKey(1)
            
            # CHỈ GỬI CLEAN_FRAME (ảnh sạch) CHO AI
            msg, success = process_check_in(clean_frame)
            status_message = msg
            status_color = (0, 255, 0) if success else (0, 0, 255)
            
        elif key == ord('r'):
            status_message = "Dang doc bien so xe RA..."
            status_color = (0, 255, 255)
            cv2.putText(frame, status_message, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow("He Thong Quan Ly Bai Xe - AI ALPR", frame)
            cv2.waitKey(1)
            
            # CHỈ GỬI CLEAN_FRAME (ảnh sạch) CHO AI
            msg, success = process_check_out(clean_frame)
            status_message = msg
            status_color = (0, 255, 0) if success else (0, 0, 255)

        elif key == ord('l'):
            records = fetch_all("SELECT license_plate_in, check_in_time FROM PARKING_SESSIONS WHERE status='Đang gửi'")
            print("\n" + "="*50)
            print(f" DANH SÁCH XE ĐANG TRONG BÃI ({len(records)} xe)")
            print("="*50)
            if not records:
                print(" Bãi xe hiện đang trống.")
            else:
                for i, (plate, time_in) in enumerate(records, 1):
                    print(f" {i}. Biển số: {plate} | Vào lúc: {time_in}")
            print("="*50 + "\n")
            
            status_message = f"Da in {len(records)} xe ra Terminal (Man hinh go lenh)!"
            status_color = (255, 255, 0) 

        elif key == ord('q'):
            print("[INFO] Đang thoát chương trình...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()