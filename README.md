NGHIÊN CỨU VÀ XÂY DỰNG HỆ THÓNG QUẢN LÝ BÃI GỬI XE BÁN TƯ ĐỘNG DỰA TRÊN IOT VÀ MÔ HÌNH HỌC SÂU
- đây là một dự án quản lý bãi đỗ xe sử dụng trí tuệ nhân tạo để nhận diện biển số xe
- hệ thống kết hợp yolov8 để phát hiện biển số và easyocr để trích xuất biển số ra khỏi camera, quản lý xe ra vào bằng cơ sở dữ liệu

CẤU TRÚC
btl.py: tệp mã nguồn
best.pt: mô hình yolo đã được huấn luyện để phát hiện biển số xe
yolov8.pt: mô hình yolo
parking_system.db: CSDL để lưu trữ xe vào - ra
saved_images: Thư mục lưu trữ hình ảnh xe vào - ra
readme.md: tệp hướng dẫn
 
 YÊU CẦU
 python: khuyến nghị dùng bản 3.11(64 bit (phiên bản hiện tại đang chạy))
 camera: thu nhận hình ảnh trực tiếp

 CÀI ĐẶT
 mở Terminal
 Cài đặt thư viện python 
 - cài các thư viện bắt buộc :
             pip install opencv-python ultralytics easyocr

SỬ DỤNG
- mở terminal chứa thư ục dự án:
            python btl.py

- điều khiển trên camera
v: chụp ảnh xe vào
r: chụp ảnh xe ra
l: xem danh sách xe đã vào
q: out khỏi chương trình

CÁCH HOẠT ĐỘNG
- dùng yolo để nhận diện biển số
- cắt vùng ahr chứa biển, dùng các bộ lọc của openv để khử nhiễu và làm rõ chữ
- đưa ảnh vào EasyOCR để đọc chuỗi kí tự
- nếu yolo không quét được thì hệ thống dùng OCR để quét toàn bộ khung hình