import cv2
import os

# Nhập tên thư mục
folder_name = input("Nhập tên thư mục: ")

# Tạo thư mục
if not os.path.exists(os.path.join("..", "Dataset", "FaceData", "raw", folder_name)):
    os.makedirs(os.path.join("..","Dataset", "FaceData", "raw", folder_name), exist_ok=True)

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Chụp ảnh
for i in range(100):
    ret, frame = cap.read()
    if not ret:
        break

    # Hiển thị quá trình chụp
    print(f"Đang chụp ảnh số {i+1}")

    # Lưu ảnh
    cv2.imwrite(os.path.join("..","Dataset", "FaceData", "raw", folder_name, f"{i:03d}.jpg"), frame)

# Hiển thị đường dẫn tới thư mục
print(f"Đường dẫn tới thư mục: {os.path.join('..','Dataset', 'FaceData', 'raw', folder_name)}")

# Đóng camera
cap.release()
