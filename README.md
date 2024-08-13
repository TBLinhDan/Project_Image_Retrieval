# **Project: Images Retrieval**

**Truy vấn hình ảnh (Images Retrieval) là một bài toán thuộc lĩnh vực Truy vấn thông tin (Information Retrieval).  
Với các hình ảnh được lấy từ một bộ dữ liệu hình ảnh cho trước, Project thực hiện xây dựng một chương trình trả về các hình ảnh (Images) có liên quan đến hình ảnh truy vấn đầu vào(Query)** *(có nghĩa là khi đưa vào chương trình đó một hình ảnh, nó sẽ hiển thị nhiều hình ảnh tương tự.)*

**Vậy nên Input/Output của một hệ thống truy vấn hình ảnh bao gồm:**  
• Input: Hình ảnh truy vấn Query Image và bộ dữ liệu Images Library.  
• Output: Danh sách các hình ảnh có sự tương tự đến hình ảnh truy vấn.

***Trong dự án này, chúng ta sẽ xây dựng một hệ thống truy xuất hình ảnh bằng cách sử dụng mô hình Deep Learning đã được huấn luyện trước (CLIP) để trích xuất đặc trưng của ảnh và thu được các vector đặc trưng. Sau đó, chúng ta sẽ sử dụng vector database để index, lưu trữ và truy xuất các ảnh tương tự với ảnh yêu cầu thông qua các thuật toán đo độ tương đồng.***

Dự án sẽ giới thiệu các phương pháp từ **cơ bản** đến **nâng cao** để xây dựng một hệ thống truy vấn ảnh. Chúng ta sẽ phát triển hệ thống này trên một tập dữ liệu cụ thể, với các mục tiêu chính bao gồm:  
• Xây dựng chương trình truy vấn ảnh cơ bản.  
• Phát triển chương trình truy vấn ảnh nâng cao với CLIP model và vector database.  
• Thu thập và xử lý dữ liệu nhằm mục đích xây dựng chương trình truy vấn ảnh cá nhân hóa.  



## **I. Chương Trình Truy Vấn Ảnh Cơ Bản:**  
**1. Tải tập dữ liệu ảnh và giải nén**  
Trên Google Colab, khởi tạo một code cell sử dụng lệnh:  
```
!gdown 1msLVo0g0LFmL9-qZ73vq9YEVZwbzOePF
!unzip data
```
(refresh lại phần Files của Google Colab để xem thư mục **data** đã xuất hiện hay chưa: trong data sẽ xuất hiện 2 class ảnh cho **train** *(nơi sẽ trả về kết
quả truy vấn)* và **test** *(nơi chứa ảnh sẽ được đem đi truy vấn)*).

**2. Import một số thư viện cần thiết:**  
Để đọc ảnh chúng ta sử dụng thư viện **PIL**; để xử lí ma trận chúng ta dử dụng **numpy**; để thao tác với thư mục, file chúng ta sử dụng thư viện **os**; sử dụng **matplotlib** để hiển thị kết quả.
```
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
```
**3. Lấy danh sách các class của ảnh trong data**
```
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
```
**4. Để thực hiện tính toán trên các hình ảnh, chúng ta sẽ đọc ảnh, resize về kích thước chung (thì mới áp dụng được các phép đo) và chuyển đổi nó về dạng numpy:**
```
def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)
```

Định nghĩa hàm ***folder_to_images()*** để tải tất cả hình ảnh thuộc về một lớp cụ thể và trả về chúng cùng với đường dẫn tệp tương ứng của chúng  
```
def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path
```
Đinh nghĩa hàm ***plot_results()***, hàm này sẽ trả về kết quả những ảnh có giá trị tương đồng theo danh sách truy vẫn tương ứng từng độ đo ***absolute_difference***, ***mean_square_difference***, ***Cosine Similarity***, ***Correlation Coefficient*** .
```
def plot_results(querquery_pathy, ls_path_score, reverse):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(querquery_pathy, size=(448,448)))
    plt.title(f"Query Image: {querquery_pathy.split('/')[2]}", fontsize=16)
    plt.axis("off")
    for i, path in enumerate(sorted(ls_path_score, key=lambda x : x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448,448)))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis("off")
    plt.show()
```

**5. Truy vấn hình ảnh với độ đo absolute_difference L1:**  
Xây dựng hàm ***absolute_difference()*** tính độ tương đồng giữa các hình ảnh.
```
def absolute_difference(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)
```
Tạo hàm ***get_l1_score()*** thực hiện tính toán độ tương đồng giữa ảnh input và các hình ảnh trong bộ dữ liệu. Hàm này sẽ trả về ảnh ***query*** và ***ls_path_score*** chứa danh sách hình ảnh và giá trị độ tương đồng với từng ảnh.
```
def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            rates = absolute_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score
```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***, đem so sánh với các hình ảnh trong tập huấn luyện ***train*** để tính
điểm ***L1***. Sau đó, kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo L1. Với độ đo L1 này thì giá trị càng nhỏ sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = False. Cuối cùng, 5 kết quả tốt nhất sẽ được hiển thị cùng với ảnh truy vấn. 
```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_l1_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=False)
```

**6. Truy vấn hình ảnh với độ đo mean_square_difference L2:**  
Xây dựng hàm ***mean_square_difference()*** tính độ tương đồng giữa các hình ảnh.
```
def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    return np.mean((data - query)**2, axis=axis_batch_size)
```
Tạo hàm ***get_l2_score()*** thực hiện tính toán độ tương đồng giữa ảnh input và các hình ảnh trong bộ dữ liệu. Hàm này sẽ trả về ảnh ***query*** và ***ls_path_score*** chứa danh sách hình ảnh và giá trị độ tương đồng với từng ảnh.
```
def get_l2_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            rates = mean_square_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score
```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***, đem so sánh với các hình ảnh trong tập huấn luyện ***train*** để tính
điểm ***L2***. Sau đó, kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo L2. Với độ đo L2 này thì giá trị càng nhỏ sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = False. Cuối cùng, 5 kết quả tốt nhất sẽ được hiển thị cùng với ảnh truy vấn. 
```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_l2_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=False)
```

**7. Truy vấn hình ảnh với độ đo Cosine Similarity:**  
Xây dựng hàm ***Cosine Similarity()*** tính độ tương đồng giữa các hình ảnh.
```
def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    # Ứng dụng norm
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)
```
Tạo hàm ***get_cosine_similarity_score()*** thực hiện tính toán độ tương đồng giữa ảnh input và các hình ảnh trong bộ dữ liệu. Hàm này sẽ trả về ảnh ***query*** và ***ls_path_score*** chứa danh sách hình ảnh và giá trị độ tương đồng với từng ảnh.
```
def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            rates = cosine_similarity(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score
```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***, đem so sánh với các hình ảnh trong tập huấn luyện ***train*** để tính
điểm ***cosine_similarity***. Sau đó, kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo ***cosine_similarity_score***. Cuối cùng, để hiển thị 5 kết quả tốt nhất chúng ta sử dụng hàm plot_results(), tuy nhiên ở hàm này chúng ta sẽ sắp xếp giá trị giảm dần từ lớn đến nhỏ vì với độ đo này thì giá trị càng lớn sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = True.

```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=True)
```

**8. Truy vấn hình ảnh với độ đo  Correlation Coefficient:**  
Xây dựng hàm *** Correlation Coefficient()*** tính độ tương đồng giữa các hình ảnh.
```
def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))

    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)
```
Tạo hàm ***get_correlation_coefficient_score()*** thực hiện tính toán độ tương đồng giữa ảnh input và các hình ảnh trong bộ dữ liệu. Hàm này sẽ trả về ảnh ***query*** và ***ls_path_score*** chứa danh sách hình ảnh và giá trị độ tương đồng với từng ảnh.
```
def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            rates = correlation_coefficient(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score
```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***, đem so sánh với các hình ảnh trong tập huấn luyện ***train*** để tính
điểm ***Correlation Coefficient***. Sau đó, kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo ***correlation_coefficient_score***. Cuối cùng, để hiển thị 5 kết quả tốt nhất chúng ta sử dụng hàm plot_results(), tuy nhiên ở hàm này chúng ta sẽ sắp xếp giá trị giảm dần từ lớn đến nhỏ vì với độ đo này thì giá trị càng lớn sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = True.

```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=True)
```

## **II. Chương Trình truy vấn Hình ảnh (nâng cao)**  
Sử dụng **Pretrained Deep Learning Model** trích xuất feature vector cho các ảnh để tăng cường khả năng truy xuất hình ảnh chính xác hơn. Khi một hình ảnh truy vấn được đưa vào mô hình, mô hình sẽ tính toán đặc trưng của hình ảnh truy vấn và so sánh chúng với các đặc trưng đã được tính toán trước của những hình ảnh được lưu trữ trên hệ thống. Sự tương đồng giữa các đặc trưng này được sử dụng để xác định các hình ảnh có liên quan nhất, và kết quả là những hình ảnh tương tự nhất với hình ảnh truy vấn được
trả về cho người dùng.

**1. Tải tập dữ liệu ảnh và giải nén**  
Trên Google Colab, khởi tạo một code cell sử dụng lệnh:  
```
!gdown 1msLVo0g0LFmL9-qZ73vq9YEVZwbzOePF
!unzip data
```
(refresh lại phần Files của Google Colab để xem thư mục **data** đã xuất hiện hay chưa: trong data sẽ xuất hiện 2 class ảnh cho **train** *(nơi sẽ trả về kết
quả truy vấn)* và **test** *(nơi chứa ảnh sẽ được đem đi truy vấn)*).

**2. Cài đặt hai thư viện quan trọng là chromadb và open-clip-torch.**
* Thư viện ***chromadb*** hỗ trợ việc quản lý và truy xuất dữ liệu hình ảnh hiệu quả (chúng ta cũng sử dụng thêm với mục đích tạo vector database)
* Và chromadb có thể dùng ***open-clip-torch*** để cung cấp khả năng sử dụng mô hình CLIP đã được đào tạo sẵn, đây là một công cụ mạnh mẽ để phân tích nội dung hình ảnh thông qua học sâu. 

```
%pip install chromadb
%pip install open-clip-torch
```

**3. Import một số thư viện cần thiết:**  
Để đọc ảnh chúng ta sử dụng thư viện **PIL**; để xử lí ma trận chúng ta dử dụng **numpy**; để thao tác với thư mục, file chúng ta sử dụng thư viện **os**; sử dụng **matplotlib** để hiển thị kết quả.
```
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
```

**4. Lấy danh sách các class của ảnh trong data**
```
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
```

**5. Khởi tạo một hàm để trích xuất vector đặc trưng từ một hình sử dụng mô hình CLIP.**  
Mô hình CLIP sẽ được sử dụng để biến đổi hình ảnh thành các vector đặc
trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Sau đó, việc so sánh các hình ảnh không được thực hiện trực tiếp trên ảnh gốc mà là thông qua việc tính sự tương đồng giữa các vector này.

```
embedding_function = OpenCLIPEmbeddingFunction()

def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)
```

**6. Đọc ảnh, resize về kích thước chung và chuyển đổi nó về dạng numpy:**
```
def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)
```

Định nghĩa hàm ***folder_to_images()*** để tải tất cả hình ảnh thuộc về một lớp cụ thể và trả về chúng cùng với đường dẫn tệp tương ứng của chúng  
```
def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path
```
Đinh nghĩa hàm ***plot_results()***, hàm này sẽ trả về kết quả những ảnh có giá trị tương đồng theo danh sách truy vẫn tương ứng từng độ đo ***absolute_difference***, ***mean_square_difference***, ***Cosine Similarity***, ***Correlation Coefficient*** .
```
def plot_results(querquery_pathy, ls_path_score, reverse):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(querquery_pathy, size=(448,448)))
    plt.title(f"Query Image: {querquery_pathy.split('/')[2]}", fontsize=16)
    plt.axis("off")
    for i, path in enumerate(sorted(ls_path_score, key=lambda x : x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448,448)))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis("off")
    plt.show()
```

**6.1 Truy vấn hình ảnh với độ đo absolute_difference L1:**  
Xây dựng hàm ***absolute_difference()*** tính độ tương đồng giữa các vector đặc trưng của các hình ảnh.
```
def absolute_difference(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)
```
Truy vấn ***embedding vector*** với độ đo L1 hàm ***get_l1_score*** được nâng cấp lên bằng cách sử dụng CLIP model để trích xuất vector đặc trưng.

```
def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = absolute_difference(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***. Thông qua Mô hình CLIP, hình ảnh thành sẽ được biến đổi thành các vector đặc trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Sau đó, tính sự tương đồng ***L1*** giữa các vector này với các hình ảnh trong tập huấn luyện ***train***. Kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo L1. Với độ đo L1 này thì giá trị càng nhỏ sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = False. Cuối cùng, 5 kết quả tốt nhất sẽ được hiển thị cùng với ảnh truy vấn. 
```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_l1_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=False)
```

**6.2 Truy vấn hình ảnh với độ đo mean_square_difference L2:**  
Xây dựng hàm ***mean_square_difference()*** tính độ tương đồng giữa các vector đặc trưng của các hình ảnh.
```
def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    return np.mean((data - query)**2, axis=axis_batch_size)
```
Truy vấn ***embedding vector*** với độ đo L2 hàm ***get_l2_score*** được nâng cấp lên bằng cách sử dụng CLIP model để trích xuất vector đặc trưng.
```
def get_l2_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = mean_square_difference(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score
```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***. Thông qua Mô hình CLIP, hình ảnh thành sẽ được biến đổi thành các vector đặc trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Sau đó, tính sự tương đồng ***L2*** giữa các vector này với các hình ảnh trong tập huấn luyện ***train***. Kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo L2. Với độ đo L2 này thì giá trị càng nhỏ sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = False. Cuối cùng, 5 kết quả tốt nhất sẽ được hiển thị cùng với ảnh truy vấn.  
```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_l2_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=False)
```

**6.3 Truy vấn hình ảnh với độ đo Cosine Similarity:**  
Xây dựng hàm ***Cosine Similarity()*** tính độ tương đồng giữa các vector đặc trưng của các hình ảnh.
```
def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))
    # Ứng dụng norm
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)
```
Truy vấn ***embedding vector*** với độ đo ***Cosine Similarity*** hàm get_cosine_similarity_score được nâng cấp lên bằng cách sử dụng CLIP model để trích xuất vector đặc trưng.
```
def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = cosine_similarity(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score
```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***. Thông qua Mô hình CLIP, hình ảnh thành sẽ được biến đổi thành các vector đặc trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Sau đó, tính sự tương đồng ***cosine_similarity*** giữa các vector này với các hình ảnh trong tập huấn luyện ***train***. Kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo ***cosine_similarity_score***. Cuối cùng, để hiển thị 5 kết quả tốt nhất chúng ta sử dụng hàm plot_results(), tuy nhiên ở hàm này chúng ta sẽ sắp xếp giá trị giảm dần từ lớn đến nhỏ vì với độ đo này thì giá trị càng lớn sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = True.

```
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=True)
```

**6.4 Truy vấn hình ảnh với độ đo  Correlation Coefficient:**  
Xây dựng hàm *** Correlation Coefficient()*** tính độ tương đồng giữa các vector đặc trưng của các hình ảnh.
```
def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1,len(data.shape)))

    # Ứng dụng mean
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)

    # Ứng dụng norm
    query_norm = np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))

    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)
```
Truy vấn ***embedding vector*** với độ đo ***Correlation Coefficient*** hàm get_correlation_coefficient_score được nâng cấp lên bằng cách sử dụng CLIP model để trích xuất vector đặc trưng.
```
def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = correlation_coefficient(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

```
Ví dụ: Hình ảnh quả Cam truy vấn ***query*** lấy trong tập ***test***, được thay đổi về cùng kích thước ***size = (448, 448)***. Thông qua Mô hình CLIP, hình ảnh thành sẽ được biến đổi thành các vector đặc trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Sau đó, tính sự tương đồng ***Correlation Coefficient*** giữa các vector này với các hình ảnh trong tập huấn luyện ***train***. Kết quả truy vấn được trả về là danh sách các đường dẫn chứa hình ảnh và điểm số tính theo ***correlation_coefficient_score***. Cuối cùng, để hiển thị 5 kết quả tốt nhất chúng ta sử dụng hàm plot_results(), tuy nhiên ở hàm này chúng ta sẽ sắp xếp giá trị giảm dần từ lớn đến nhỏ vì với độ đo này thì giá trị càng lớn sẽ càng giống nhau, cho nên chúng ta sử dụng reverse = True.

```
root_img_path = "data/train/"
query_path = "data/test/Orange_easy/0_100.jpg"
size = (448, 448)
query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=True)
```



## **III.  Tối ưu hoá quá trình truy vấn hình ảnh sử dụng mô hình CLIP và cơ sở dữ liệu vector**  
Vì mỗi lần truy vấn đều cần phải sử dụng lại mô hình CLIP, phương pháp này sẽ sử dụng một cơ sở dữ liệu vector (vector database) để quản lý các embedding vector, giúp quá trình truy vấn được tối ưu hơn.

**1. Tải tập dữ liệu ảnh và giải nén**  
Trên Google Colab, khởi tạo một code cell sử dụng lệnh:  
```
!gdown 1msLVo0g0LFmL9-qZ73vq9YEVZwbzOePF
!unzip data
```
(refresh lại phần Files của Google Colab để xem thư mục **data** đã xuất hiện hay chưa: trong data sẽ xuất hiện 2 class ảnh cho **train** *(nơi sẽ trả về kết
quả truy vấn)* và **test** *(nơi chứa ảnh sẽ được đem đi truy vấn)*).

**2. Cài đặt hai thư viện quan trọng là chromadb và open-clip-torch.**
* Thư viện ***chromadb*** hỗ trợ việc quản lý và truy xuất dữ liệu hình ảnh hiệu quả (chúng ta cũng sử dụng thêm với mục đích tạo vector database)
* Và chromadb có thể dùng ***open-clip-torch*** để cung cấp khả năng sử dụng mô hình CLIP đã được đào tạo sẵn, đây là một công cụ mạnh mẽ để phân tích nội dung hình ảnh thông qua học sâu.

```
%pip install chromadb
%pip install open-clip-torch
```

**3. Import một số thư viện cần thiết:**  
Để đọc ảnh chúng ta sử dụng thư viện **PIL**; để xử lí ma trận chúng ta dử dụng **numpy**; để thao tác với thư mục, file chúng ta sử dụng thư viện **os**; sử dụng **matplotlib** để hiển thị kết quả.
```
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
```

**4. Tạo list các đường dẫn cho ảnh lấy embedding và được đưa vào database**  
```
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
HNSW_SPACE = "hnsw:space"
```
Định nghĩa Hàm ***get_files_path*** trích xuất đường dẫn của các ảnh từ một thư mục cho trước. Đầu tiên, chúng ta sẽ liệt kê các thư mục con dựa trên tên của các class (CLASS_NAME). Sau đó, liệt kê tất cả các ảnh trong mỗi thư mục con và lưu trữ đường dẫn của từng ảnh vào một danh sách. Có được danh sách đường dẫn của các ảnh rồi, ta mới trích xuất vector đặc trưng từ các ảnh và lưu trữ chúng vào cơ sở dữ liệu được

```
def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + '/' + filename
            files_path.append(filepath)
    return files_path
```
```
data_path = f'{ROOT}/train'
files_path = get_files_path(path=data_path)
files_path
```
Đinh nghĩa hàm ***plot_results()***, hàm này sẽ trả về kết quả những ảnh có giá trị tương đồng theo danh sách truy vẫn tương ứng từng độ đo .
```
def plot_results(image_path, files_path, results):
    query_image = Image.open(image_path).resize((448,448))
    images = [query_image]
    class_name = []
    for id_img in results['ids'][0]:
        id_img = int(id_img.split('_')[-1])
        img_path = files_path[id_img]
        img = Image.open(img_path).resize((448,448))
        images.append(img)
        class_name.append(img_path.split('/')[2])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Iterate through images and plot them
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if i == 0:
            ax.set_title(f"Query Image: {image_path.split('/')[2]}")
        else:
            ax.set_title(f"Top {i+1}: {class_name[i-1]}")
        ax.axis('off')  # Hide axes
    # Display the plot
    plt.show()
```

**5. Image Embedding:**  
Khởi tạo một hàm để trích xuất vector đặc trưng từ một hình sử dụng mô hình CLIP. Mô hình CLIP sẽ được sử dụng để biến đổi hình ảnh thành các vector đặc
trưng đại diện cho nội dung và ngữ cảnh của hình ảnh đó. Sau đó, việc so sánh các hình ảnh không được thực hiện trực tiếp trên ảnh gốc mà là thông qua việc tính sự tương đồng giữa các vector này.

```
embedding_function = OpenCLIPEmbeddingFunction()

def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=np.array(image))
    return embedding
```
```
img = Image.open('data/train/African_crocodile/n01697457_260.JPEG')
get_single_image_embedding(image=img)
```

**6. Chromadb L2 Embedding Collection:**  
Truy vấn ảnh với L2 Collection Trong ChromaDB, "collection" là một khái niệm quan trọng, dùng
để tổ chức và quản lý dữ liệu. Một collection trong ChromaDB có thể được hiểu như là một tập hợp các
vector hoặc tài liệu được chỉ mục và lưu trữ cùng nhau dựa trên một số tiêu chí hoặc đặc điểm chung.
Nó tương tự như concept của "table" trong cơ sở dữ liệu quan hệ hoặc "collection" trong MongoDB.
Đoạn code sau đây định nghĩa hàm add_embedding, một hàm giúp trích xuất và lưu trữ các vector
đặc trưng của ảnh vào một collection đã được tạo.

```
def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding)
    collection.add(
        embeddings=embeddings,
        ids=ids
    )
```

Khởi tạo một client cho cơ sở dữ liệu Chroma và tạo một collection mới với cấu hình
sử dụng L2 để so sánh các embedding vector. Sau đó, gọi hàm add_embedding để thêm các vector đặc
trưng của ảnh vào collection này, qua đó tạo điều kiện thuận lợi cho việc truy vấn nhanh chóng và hiệu
quả.
```
# Create a Chroma Client
chroma_client = chromadb.Client()
# Create a collection
l2_collection = chroma_client.get_or_create_collection(name="l2_collection",
                                                           metadata={HNSW_SPACE: "l2"})
add_embedding(collection=l2_collection, files_path=files_path)
```

**7. Search Image With L2 Collection:**  
Hàm search được định nghĩa để thực hiện truy xuất các ảnh dựa trên embedding của ảnh truy vấn.
Hàm này nhận đường dẫn của ảnh truy vấn, loại collection và số lượng kết quả trả về mong muốn, sau đó trả về danh sách các kết quả phù hợp.
```
def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results # how many results to return
    )
    return results
```
```
test_path = f'{ROOT}/test'
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[1]
l2_results = search(image_path=test_path, collection=l2_collection, n_results=5)

l2_results
```
```
plot_results(image_path=test_path, files_path=files_path, results=l2_results)
```

**8. Search Image With Cosine similarity Collection**

```
# Create a collection
cosine_collection = chroma_client.get_or_create_collection(name="Cosine_collection",
                                                           metadata={HNSW_SPACE: "cosine"})
add_embedding(collection=cosine_collection, files_path=files_path)
```
```
test_path = f'{ROOT}/test'
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[3]
cosine_results = search(image_path=test_path, collection=cosine_collection, n_results=5)

cosine_results
```
```
plot_results(image_path=test_path, files_path=files_path, results=cosine_results)
```

