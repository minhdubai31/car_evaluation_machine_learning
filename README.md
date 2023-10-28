## Các thư viện cần có:
    scikit-learn
    pandas
    numpy
    matplotlib
    seaborn
    flask
    flask_cors

> (Nếu thiếu) Có thể install bằng lệnh:    `pip install -U <tên thư viện>`

**********************************************************************************
## Cấu trúc project:

 - Thư mục ./resource chứa dataset (file CSV)
 - Thư mục ./img chứa các biểu đồ mà code sẽ sinh ra, mỗi thư mục con chứa 
    một dạng biểu đồ
 - File main.py sẽ đọc, xử lý dữ liệu và sinh ra biểu đồ lưu vào thư mục ./img
 - Thư mục ./app:
	 - File ./public/index.html là một giao diện web nho nhỏ, cho phép đánh giá 
        xe, người dùng có thể dùng form hoặc upload file (phải chạy file server.py 
        lên mới dùng đánh giá được)
	- File server.py dùng để chạy backend, xử lý yêu cầu từ web
	- File randomforest_model.joblib là cái model mà tôi đã train sẵn
	- File tailwind.config.js là để tôi dùng tailwindcss code giao diện web dễ
        nhìn một chút
