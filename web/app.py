import streamlit as st
st.set_page_config(page_title="Multi-aspect Comment Classifier", page_icon=":bar_chart:", layout="wide")

import pandas as pd
import numpy as np
import os
import joblib
import re
import emoji
# from num2words import num2words # Không được sử dụng, có thể bỏ
# from deep_translator import GoogleTranslator # Không được sử dụng, có thể bỏ
import matplotlib.pyplot as plt
# import seaborn as sns # Không được sử dụng, có thể bỏ
from wordcloud import WordCloud
# from grouped_wordcloud import GroupedColorFunc # Không được sử dụng, có thể bỏ
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from time import sleep
import random
from pathlib import Path
# from joblib import dump, load # joblib đã được import ở trên
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer
from gensim.models import Word2Vec
import tensorflow as tf # Để tải mô hình Keras

os.environ["STREAMLIT_WATCHED_MODULES"] = "false" # Nên đặt ở đầu file

# ==== CẤU HÌNH ĐƯỜNG DẪN ====
# Sử dụng Path cho đường dẫn để tương thích đa nền tảng tốt hơn
BASE_APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_APP_DIR / "model" # Ví dụ: nếu thư mục model ngang hàng với file script
DATA_DIR = BASE_APP_DIR / "data" # Ví dụ: nếu thư mục data ngang hàng với file script

# Nếu bạn muốn giữ đường dẫn tuyệt đối từ user:
# model_dir = r"C:\Users\84913\Downloads\TranVanLoc_HoLeKhoiNguyen\model"
# data_base_dir = r"C:\Users\84913\Downloads\TranVanLoc_HoLeKhoiNguyen\data"

# Sử dụng MODEL_DIR và DATA_DIR đã định nghĩa ở trên sẽ linh hoạt hơn
# Ví dụ:
# model_dir = str(MODEL_DIR) # Chuyển Path object thành string nếu cần
# abbs_path = DATA_DIR / "abbreviations.xlsx"
# stopwords_path = DATA_DIR / "vietnamesestopwords.txt"

# Tạm thời giữ lại đường dẫn tuyệt đối từ người dùng để khớp với code gốc
model_dir = r"E:\practice_with_fren\khoinguyen\model"
user_data_dir = r"E:\practice_with_fren\khoinguyen\data" # Để tải abbreviation và stopwords

model_files = {
    "price": "PRICE_MODEL.joblib", # Giả sử PRICE_MODEL là Keras model, tên file có thể là .h5 hoặc thư mục
    "quality": "QUALITY_MODEL.joblib",
    "service": "SERVICE_MODEL.joblib", # Giả sử SERVICE_MODEL là Keras model
    "store": "STORE_MODEL.joblib",
    "packaging": "PACKAGING_MODEL.joblib",
    "others": "OTHERS_MODEL.joblib", # Thêm dấu phẩy
}

# Create a separate variable for uppercase keys
model_files_keys_uppercase = [k.upper() for k in model_files.keys()]

models = {}
for label, fname in model_files.items():
    path = os.path.join(model_dir, fname)
    try:
        # Giả định rằng PRICE và SERVICE là các mô hình Neural Network (Keras)
        # và các mô hình khác là scikit-learn (tải bằng joblib)
        # Bạn cần điều chỉnh tên file hoặc cách tải cho phù hợp (ví dụ: .h5 cho Keras)
        if label in ["price", "service"] and (fname.endswith(".h5") or os.path.isdir(path)): # Kiểm tra nếu là file .h5 hoặc thư mục SavedModel
            models[label] = tf.keras.models.load_model(path)
            print(f"→ Loaded Keras model {label} from {path}")
        else:
            models[label] = joblib.load(path)
            print(f"→ Loaded {label} via joblib.load from {path}")
    except Exception as e:
        print(f"[ERROR] Loading model {label} from {path}: {e}")

# Load Keras Tokenizer (đã được fit và lưu lại)
# Giả sử tokenizer.pkl là Keras Tokenizer đã lưu
try:
    tokenizer_load = joblib.load(os.path.join(model_dir, "tokenizer.pkl"))
    print("→ Keras Tokenizer loaded successfully.")
except Exception as e:
    print(f"[ERROR] Loading Keras Tokenizer: {e}")
    tokenizer_load = None # Đặt là None nếu không tải được

# Load Word2Vec model
w2v_model_path = os.path.join(model_dir, "word2vec_sentiment.model")
try:
    w2v_model = Word2Vec.load(w2v_model_path)
    print("→ Word2Vec model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Loading Word2Vec model from {w2v_model_path}: {e}")
    w2v_model = None

print("All models & tokenizer should be ready (check for errors above)!")

# ==== LOAD RESOURCES (Abbreviations and Stopwords) ====
# Sử dụng đường dẫn đã khai báo ở phần "CẤU HÌNH ĐƯỜNG DẪN"
# Nếu dùng BASE_APP_DIR:
# abbs_path = DATA_DIR / "abbreviations.xlsx"
# stopwords_path = DATA_DIR / "vietnamesestopwords.txt"
# Nếu dùng đường dẫn tuyệt đối của người dùng:
abbs_path = os.path.join(user_data_dir, "abbreviations.xlsx")
stopwords_path = os.path.join(user_data_dir, "vietnamesestopwords.txt")


print(f"Loading abbreviations from: {abbs_path}")
try:
    abbs_df = pd.read_excel(abbs_path, engine='openpyxl') # engine='openpyxl' nếu là file .xlsx
    abbreviation_dict = dict(zip(abbs_df['abbreviation'].astype(str), abbs_df['meaning']))
except Exception as e:
    print(f"[ERROR] Loading abbreviations: {e}")
    abbreviation_dict = {}

print(f"Loading stopwords from: {stopwords_path}")
try:
    with open(stopwords_path, "r", encoding="utf-8") as f:
        custom_stopwords = set(f.read().splitlines()) # splitlines() an toàn hơn
except Exception as e:
    print(f"[ERROR] Loading stopwords: {e}")
    custom_stopwords = set()


# ==== CRAWL FROM HASAKI ====
# Get rating for comments:
def get_star(string):
    start_index = string.find(':')
    end_index = string.find('%')
    if start_index != -1 and end_index != -1 and end_index > start_index + 1:
        try:
            return int(string[start_index+1:end_index]) / 20
        except ValueError:
            return 0 # Hoặc giá trị mặc định khác
    return 0


def crawl_comments_from_link(input_link_button):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--headless") # Chạy ẩn trình duyệt
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")


    driver = None # Khởi tạo driver là None
    try:
        driver = webdriver.Chrome(options=options)
        sleep(random.uniform(1,3)) # uniform cho số thực
        
        name_comment_all, content_comment_all, product_variant_all, datetime_comment_all, rating_comment_all = [], [], [], [], []
        
        driver.get(input_link_button)
        sleep(random.uniform(5,7))
        
        elem = driver.find_element(By.CSS_SELECTOR, 'meta[name="product:retailer_item_id"]')
        product_id = elem.get_attribute('content')
        
        elems_data_productids_list = driver.find_elements(By.CSS_SELECTOR, 'div.w-full.flex.flex-wrap.gap-2\\.5.text-xs a.relative')
        uniq_data_productids_list = [elem.get_attribute('href') for elem in elems_data_productids_list]
        product_ids_list = [re.search(r'-(\d+)\.html', url).group(1) for url in uniq_data_productids_list if re.search(r'-(\d+)\.html', url)]
        
        elems_cmtpage_nums = driver.find_elements(By.CSS_SELECTOR, '.pagination_comment a')
        if elems_cmtpage_nums:
            commentpage_nums = [int(elem.get_attribute('rel')) for elem in elems_cmtpage_nums
                                if elem.get_attribute('rel') and elem.get_attribute('rel').isdigit()]
            max_cmtpage = max(commentpage_nums) if commentpage_nums else 1
        else:
            max_cmtpage = 1

        for page_num in range(1, max_cmtpage + 1):
            try:
                sleep(random.uniform(2,3))
                print(f"Crawl Page {page_num}/{max_cmtpage}")
                
                current_names = [elem.text for elem in driver.find_elements(By.CSS_SELECTOR , ".mt-2\\.5 .pb-2\\.5 .font-bold")]
                name_comment_all.extend(current_names)

                elems_content = driver.find_elements(By.CSS_SELECTOR, ".mt-\\[5px\\]")
                new_comments_on_page = []
                for elem in elems_content:
                    try:
                        full_text = elem.text.strip()
                        # Cố gắng loại bỏ phần trả lời của quản trị viên (nếu có)
                        # Selector này có thể cần điều chỉnh tùy theo cấu trúc HTML thực tế của Hasaki
                        reply_elems = elem.find_elements(By.CSS_SELECTOR, ".mt-2\\.5.pl-\\[34px\\]") # Ví dụ selector cho phần reply
                        reply_text = ""
                        if reply_elems:
                            reply_text = reply_elems[0].text.strip()
                        
                        comment_only = full_text.replace(reply_text, '').strip() if reply_text else full_text
                    except NoSuchElementException: # Nếu không có phần reply
                        comment_only = elem.text.strip()
                    
                    if comment_only: # Chỉ thêm nếu bình luận không rỗng
                        new_comments_on_page.append(comment_only)
                content_comment_all.extend(new_comments_on_page)
                
                current_variants = [elem.text for elem in driver.find_elements(By.CSS_SELECTOR , "div.mt-2\\.5 div.flex.items-center.gap-2 div.text-\\[\\#999\\]")]
                product_variant_all.extend(current_variants)
                
                current_datetimes = [elem.text for elem in driver.find_elements(By.CSS_SELECTOR , "div.mt-2\\.5 div.flex.items-center.gap-2 div.text-\\[\\#666\\]")]
                datetime_comment_all.extend(current_datetimes)

                current_ratings = [get_star(elem.get_attribute('style')) for elem in driver.find_elements(By.CSS_SELECTOR , "div.mt-2\\.5 div.flex.items-center.gap-2 div.relative.flex.items-center div.absolute")]
                rating_comment_all.extend(current_ratings)
                
                if page_num < max_cmtpage:
                    next_pagination_cmt = driver.find_element(By.XPATH, "//button[contains(@class, 'ml-[-3px]') and not(@disabled)]") # Tìm nút next không bị disabled
                    actions = ActionChains(driver)
                    actions.move_to_element(next_pagination_cmt).click().perform()
                    print("Clicked on button next page!")
                    sleep(random.uniform(2,3))
                else:
                    print("Last page reached.")
                    break

            except ElementNotInteractableException:
                print("Element Not Interactable Exception on page " + str(page_num) + " ! Likely end of comments or popup.")
                break
            except NoSuchElementException:
                print("Next page button not found or other element missing on page " + str(page_num) + "!")
                break
        
        # Đảm bảo tất cả các danh sách có cùng độ dài bằng cách cắt bớt theo danh sách ngắn nhất (thường là content_comment)
        min_len = min(
            len(name_comment_all), 
            len(content_comment_all), 
            len(product_variant_all), 
            len(datetime_comment_all), 
            len(rating_comment_all)
        )
        comment_data = pd.DataFrame({
            'name_comment': name_comment_all[:min_len], 
            'content_comment': content_comment_all[:min_len],
            'product_variant': product_variant_all[:min_len], 
            'datetime_comment': datetime_comment_all[:min_len], 
            'rating': rating_comment_all[:min_len]
        })  
        
        comment_data.insert(0, "link_item", input_link_button)
        comment_data.insert(1, "data_product_id_list", [product_ids_list] * len(comment_data))
        comment_data.insert(2, "data_product_id", product_id)
        
        sleep(random.uniform(1,2))
        return comment_data

    except Exception as e:
        st.error(f"Lỗi trong quá trình crawl: {e}")
        if driver:
            driver.quit()
        return pd.DataFrame() # Trả về DataFrame rỗng nếu có lỗi
    finally:
        if driver:
            driver.quit()


# ==== TEXT PREPROCESSING ====
def preprocess_text(text, current_abbreviation_dict, current_stopwords):
    if not isinstance(text, str):
        return [] # Trả về list rỗng nếu không phải string
    text = text.lower()
    for abbreviation, meaning in current_abbreviation_dict.items():
        text = re.sub(r'\b' + re.escape(str(abbreviation)) + r'\b', str(meaning), text) # Đảm bảo abbreviation và meaning là string
    text = emoji.demojize(text, language='en') # Thêm language='vi' nếu có thể
    text = re.sub(r"[!@#$\[\]()]", "", text) # Loại bỏ các ký tự này
    text = re.sub(r'[^\w\s_]', '', text) # Giữ lại dấu gạch dưới vì ViTokenizer có thể cần
    tokenized_text = ViTokenizer.tokenize(text)
    tokens = [word for word in tokenized_text.split() if word not in current_stopwords]
    return tokens


# ==== VECTORIZE & PREDICT ====
# Hàm chuyển đổi dữ liệu sang vector đặc trưng bằng Word2Vec
def vectorize_comment_w2v(comment_tokens, w2v_model_instance, vector_dim=150):
    if not w2v_model_instance: # Kiểm tra nếu w2v_model không được tải
        return np.zeros(vector_dim)
    
    words = comment_tokens # comment_tokens đã là list các từ
    word_vectors = [w2v_model_instance.wv[word] for word in words if word in w2v_model_instance.wv.index_to_key]
    if not word_vectors: # Sửa: len(word_vectors) == 0 thành not word_vectors
        return np.zeros(vector_dim)
    return np.mean(word_vectors, axis=0)

# Ánh xạ kết quả dự đoán (số) sang nhãn (text)
# BẠN CẦN TÙY CHỈNH CHO PHÙ HỢP VỚI MÔ HÌNH CỦA BẠN
# Ví dụ: {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
# Hoặc {0: "Không hài lòng", 1: "Hài lòng"}
# Hoặc {0: "Không đề cập", 1: "Tiêu cực", 2: "Trung tính", 3: "Tích cực"}
# Phải đảm bảo mapping này khớp với cách model của bạn được huấn luyện
reverse_label_mapping = {
    0: "Không đề cập / Trung tính", # Hoặc bất kỳ nhãn nào tương ứng với output 0
    1: "Tiêu cực",
    2: "Tích cực"
    # Thêm các mapping khác nếu cần, ví dụ:
    # 3: "Rất tích cực"
}
# Nếu model của bạn chỉ có 2 class (0: Tiêu cực, 1: Tích cực) thì sửa lại:
# reverse_label_mapping = { 0: "Tiêu cực", 1: "Tích cực" }


def predict_sentiment_for_comment(input_text):
    if not input_text.strip(): # Nếu bình luận rỗng
        # Trả về kết quả mặc định (ví dụ: "Không đề cập / Trung tính" cho tất cả các khía cạnh)
        default_sentiment = reverse_label_mapping.get(0, "N/A")
        return {aspect: default_sentiment for aspect in model_files_keys_uppercase}

    # Tiền xử lý chung
    processed_tokens = preprocess_text(input_text, abbreviation_dict, custom_stopwords)

    # Vector hóa bằng Word2Vec cho các model truyền thống (RF, SVM,...)
    # Giả sử vector_dim của Word2Vec model là 150
    input_vector_w2v = vectorize_comment_w2v(processed_tokens, w2v_model, vector_dim=150) 
    input_vector_w2v_reshaped = np.array([input_vector_w2v]) # Reshape cho scikit-learn models

    # Chuẩn bị input cho model NN (LSTM, GRU,...)
    # Ghép lại thành chuỗi duy nhất để đưa vào tokenizer
    processed_text_joined_for_nn = " ".join(processed_tokens)
    
    padded_sequence = None
    if tokenizer_load: # Chỉ thực hiện nếu tokenizer được tải thành công
        sequence = tokenizer_load.texts_to_sequences([processed_text_joined_for_nn])
        max_length = 100 # Cần khớp với max_length lúc huấn luyện model NN
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    else: # Xử lý trường hợp tokenizer không tải được
        st.warning("Keras Tokenizer chưa được tải, các mô hình NN có thể không hoạt động.")


    # Dự đoán cho từng khía cạnh
    predictions = {}

    # QUALITY (Giả sử là model scikit-learn, dùng Word2Vec)
    if 'quality' in models and models['quality']:
        quality_pred_raw = models['quality'].predict(input_vector_w2v_reshaped)[0]
        predictions["QUALITY"] = reverse_label_mapping.get(quality_pred_raw, f"Unknown: {quality_pred_raw}")
    else:
        predictions["QUALITY"] = "Model not loaded"

    # SERVICE (Giả sử là model Keras NN, dùng padded_sequence)
    if 'service' in models and models['service'] and padded_sequence is not None:
        service_pred_proba = models['service'].predict(padded_sequence)
        service_pred_raw = np.argmax(service_pred_proba, axis=1)[0]
        predictions["SERVICE"] = reverse_label_mapping.get(service_pred_raw, f"Unknown: {service_pred_raw}")
    else:
        predictions["SERVICE"] = "Model/Tokenizer not loaded"
        
    # OTHERS (Giả sử là model scikit-learn, dùng Word2Vec)
    if 'others' in models and models['others']:
        others_pred_raw = models['others'].predict(input_vector_w2v_reshaped)[0]
        predictions["OTHERS"] = reverse_label_mapping.get(others_pred_raw, f"Unknown: {others_pred_raw}")
    else:
        predictions["OTHERS"] = "Model not loaded"

    # STORE (Giả sử là model scikit-learn, dùng Word2Vec)
    if 'store' in models and models['store']:
        store_pred_raw = models['store'].predict(input_vector_w2v_reshaped)[0]
        predictions["STORE"] = reverse_label_mapping.get(store_pred_raw, f"Unknown: {store_pred_raw}")
    else:
        predictions["STORE"] = "Model not loaded"

    # PACKAGING (Giả sử là model scikit-learn, dùng Word2Vec)
    if 'packaging' in models and models['packaging']:
        packaging_pred_raw = models['packaging'].predict(input_vector_w2v_reshaped)[0]
        predictions["PACKAGING"] = reverse_label_mapping.get(packaging_pred_raw, f"Unknown: {packaging_pred_raw}")
    else:
        predictions["PACKAGING"] = "Model not loaded"

    # PRICE (Giả sử là model Keras NN, dùng padded_sequence)
    if 'price' in models and models['price'] and padded_sequence is not None:
        price_pred_proba = models['price'].predict(padded_sequence)
        price_pred_raw = np.argmax(price_pred_proba, axis=1)[0]
        predictions["PRICE"] = reverse_label_mapping.get(price_pred_raw, f"Unknown: {price_pred_raw}")
    else:
        predictions["PRICE"] = "Model/Tokenizer not loaded"
        
    # In ra kết quả (tùy chọn, có thể xóa nếu không cần log ở console)
    # print(f"--- Predictions for: '{input_text[:50]}...' ---")
    # for aspect, sentiment in predictions.items():
    #     print(f"{aspect}: {sentiment}")

    return predictions

def predict_all_labels(comment_list):
    # Khởi tạo dictionary để lưu kết quả, mỗi khía cạnh là một list các dự đoán
    # Chuyển key của model_files thành chữ hoa để nhất quán với output
    aspect_keys_uppercase = [key.upper() for key in model_files_keys_uppercase]
    all_predictions = {aspect_key: [] for aspect_key in aspect_keys_uppercase}

    if not w2v_model or not tokenizer_load:
        st.error("Word2Vec model hoặc Keras Tokenizer chưa được tải. Không thể thực hiện dự đoán.")
        # Trả về giá trị mặc định cho tất cả bình luận
        default_sentiment = reverse_label_mapping.get(0, "N/A") # Hoặc một giá trị lỗi
        for aspect_key in all_predictions.keys():
            all_predictions[aspect_key] = [default_sentiment] * len(comment_list)
        return all_predictions


    with st.spinner("Đang phân tích từng bình luận..."):
        progress_bar = st.progress(0)
        total_comments = len(comment_list)
        for i, comment_text in enumerate(comment_list):
            single_comment_predictions = predict_sentiment_for_comment(comment_text)
            for aspect_key_upper in aspect_keys_uppercase: # PACKAGING, PRICE,...
                # Tìm key gốc (thường) trong single_comment_predictions
                # (ví dụ: single_comment_predictions có key 'PRICE', 'QUALITY')
                if aspect_key_upper in single_comment_predictions:
                     all_predictions[aspect_key_upper].append(single_comment_predictions[aspect_key_upper])
                else: # Xử lý trường hợp key không khớp (ít khi xảy ra nếu logic đúng)
                    all_predictions[aspect_key_upper].append("Lỗi key")
            progress_bar.progress((i + 1) / total_comments)
    return all_predictions


# ==== STREAMLIT APP ====
st.title("🛍️ Phân tích bình luận sản phẩm từ link Hasaki")

product_link = st.text_input("🔗 Nhập link sản phẩm Hasaki:", placeholder="Ví dụ: https://hasaki.vn/san-pham/ten-san-pham-12345.html")

if 'comments_df' not in st.session_state:
    st.session_state.comments_df = pd.DataFrame()
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False


if st.button("🚀 Phân tích bình luận"):
    if product_link.strip() and ("hasaki.vn/san-pham/" in product_link or "hasaki.vn/p/" in product_link) :
        with st.spinner("⏳ Đang thu thập dữ liệu bình luận từ Hasaki... Vui lòng chờ trong giây lát."):
            comments_df_crawled = crawl_comments_from_link(product_link)
        
        if comments_df_crawled.empty:
            st.warning("❗ Không tìm thấy bình luận nào hoặc có lỗi trong quá trình crawl từ sản phẩm này.")
            st.session_state.analysis_done = False
        else:
            st.success(f"🔍 Thu thập xong! Tìm thấy {len(comments_df_crawled)} bình luận.")
            st.session_state.comments_df = comments_df_crawled
            
            comment_list = st.session_state.comments_df['content_comment'].fillna("").astype(str).tolist()
            
            if not comment_list:
                st.warning("Danh sách bình luận rỗng sau khi trích xuất.")
                st.session_state.analysis_done = False
            else:
                pred_results_dict = predict_all_labels(comment_list) # Đây là dict of lists
                
                # Gán kết quả dự đoán vào DataFrame
                for label_aspect_upper, sentiments_list in pred_results_dict.items():
                    # Đảm bảo độ dài của list sentiments khớp với số dòng DataFrame
                    if len(sentiments_list) == len(st.session_state.comments_df):
                        st.session_state.comments_df[label_aspect_upper] = sentiments_list
                    else:
                        st.error(f"Lỗi độ dài khi gán cột {label_aspect_upper}. Dự kiến {len(st.session_state.comments_df)}, nhận được {len(sentiments_list)}")
                        # Có thể gán một giá trị mặc định hoặc xử lý khác
                        st.session_state.comments_df[label_aspect_upper] = ["Lỗi độ dài"] * len(st.session_state.comments_df)

                st.session_state.analysis_done = True
    else:
        st.error("❌ Vui lòng nhập link sản phẩm Hasaki hợp lệ.")
        st.session_state.analysis_done = False

if st.session_state.analysis_done and not st.session_state.comments_df.empty:
    st.subheader("📄 Kết quả phân loại bình luận")
    # Hiển thị các cột cần thiết, bao gồm cả các cột dự đoán
    display_cols = ['content_comment', 'rating'] + [col for col in st.session_state.comments_df.columns if col.isupper() and col in model_files_keys_uppercase]
    # Tạo model_files_keys_uppercase để sử dụng ở đây
    # Ví dụ: model_files_keys_uppercase = [k.upper() for k in model_files.keys()]
    # Tạm thời dùng list các key đã biết:
    aspect_display_cols = [key.upper() for key in model_files_keys_uppercase]
    cols_to_show = ['content_comment', 'rating'] + aspect_display_cols
    st.dataframe(st.session_state.comments_df[cols_to_show])

    st.subheader("📊 Thống kê cảm xúc theo từng khía cạnh")
    # Các cột khía cạnh (đã được thêm vào df với tên viết hoa)
    label_cols_for_viz = aspect_display_cols

    # Tạo các tab cho mỗi khía cạnh
    tabs = st.tabs(label_cols_for_viz)

    for i, aspect_col_name in enumerate(label_cols_for_viz):
        with tabs[i]:
            if aspect_col_name in st.session_state.comments_df.columns:
                st.markdown(f"#### Khía cạnh: {aspect_col_name.capitalize()}")
                sentiment_counts = st.session_state.comments_df[aspect_col_name].value_counts()
                
                if not sentiment_counts.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sentiment_counts.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet'])
                    ax.set_ylabel("Số lượng bình luận")
                    ax.set_title(f"Phân bổ cảm xúc cho khía cạnh: {aspect_col_name.capitalize()}")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("Không có dữ liệu để hiển thị cho khía cạnh này.")
            else:
                 st.write(f"Không tìm thấy cột dữ liệu cho khía cạnh: {aspect_col_name}")


    st.subheader("☁️ Word Cloud từ tất cả bình luận")
    # Lấy lại danh sách bình luận từ DataFrame (nếu đã được cập nhật)
    comment_list_for_wc = st.session_state.comments_df['content_comment'].fillna("").astype(str).tolist()
    if comment_list_for_wc:
        all_processed_tokens = []
        for comment_text in comment_list_for_wc:
            tokens = preprocess_text(comment_text, abbreviation_dict, custom_stopwords)
            all_processed_tokens.extend(tokens)
        
        if all_processed_tokens:
            cleaned_text_for_wc = ' '.join(all_processed_tokens)
            try:
                # Kiểm tra font_path nếu dùng font tiếng Việt cho WordCloud
                # font_path_wc = "arial.ttf" # Thay bằng đường dẫn font phù hợp nếu cần
                wc = WordCloud(width=800, height=400, background_color='white',
                               # font_path=font_path_wc, # Bỏ comment nếu có font
                               collocations=False).generate(cleaned_text_for_wc)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            except Exception as e_wc:
                st.error(f"Lỗi tạo Word Cloud: {e_wc}")
                st.write("Nội dung đã xử lý cho Word Cloud (kiểm tra): ", cleaned_text_for_wc[:500])

        else:
            st.write("Không có từ nào để tạo Word Cloud sau khi xử lý.")
    else:
        st.write("Không có bình luận nào để tạo Word Cloud.")

# Thêm key model_files_keys_uppercase vào cuối (ví dụ)
# Đây là một cách tạm để code chạy, bạn nên quản lý keys tốt hơn
# model_files_keys_uppercase = [k.upper() for k in model_files.keys()]