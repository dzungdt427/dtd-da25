# IMPORT CAC THU VIEN
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from underthesea import word_tokenize, pos_tag, sent_tokenize # sent_tokenize de phan tich cau, khong phai phan tich van ban
import regex
import string
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

from googletrans import Translator
translator = Translator()
# # Test Translate English to Vietnamese
# translation = translator.translate("How are you today?", src='en', dest='vi')
# print("Original:", translation.origin)
# print("Translated:", translation.text)

# 1. TIEN XU LY DU LIEU TIENG VIET
# 1.1. Doc cac file
#LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


## 1.2. Bo sung, cap nhat list tu
new_teen_dict = {'ot':'overtime','oc':'overcome','ot.':'overtime','x':'nhân','hk':'hong_kong','jp':'japan','eu':'europe','sin':'singapore',
                'cv':'công_việc',
                'bhyt':'bảo_hiểm_y_tế','bhxh':"bảo_hiểm_xã_hội",'bh':'bảo_hiểm',
                'env':'environment','env.':'environment','dev':'developer','dept':'department','dept.':'department','dev.':'developer',
                 'ko':'không','k':'không','h':'giờ',
                 'wfh':'work from home','hn':'hà_nội','hcm':'hồ_chí_minh','hcmc':'hồ_chí_minh','vn':'việt_nam',
                 'pm':'project manager','nv':'nhân_viên','tg':'thời_gian','cty':'công_ty',
                 'đc':'được','dc':'được','x2':'gấp_đôi','x3':'gấp_ba','hr':'human resources',
                 'pt':'hlv_thể_hình','ks':'khách_sạn','vp':'văn_phòng','mn':'mọi_người','sv':'sinh_viên',
                 'pc':'máy_tính_bàn','dt':'điện_thoại','pv':'phỏng_vấn','vs':'và','mội':'mọi','ngh':'nghiệm','kn':'kinh_nghiệm'}
teen_dict.update(new_teen_dict)
del teen_dict['a']

new_eng_to_vn_dict = {'overtime':'làm_thêm_giờ','work from home':'làm_việc_tại_nhà','hybrid':'làm_việc_linh_hoạt','match':'phù hợp','range':'dải','even':'sự_kiện',
                      'up':'tăng','ok':'tốt','no':'không',
                      'internship':'thực_tập','intern':'thực_tập_sinh','deal':'thỏa_thuận',
                      'conpensation':'lương','supportive':'hỗ_trợ','suportive':'hỗ_trợ','skills':'kỹ_năng','profile':'hồ_sơ','skill':'kỹ_năng',
                      'benefit':'lợi_ích','salary':'lương',
                      'overcome':'vượt_qua_khó_khăn','project manager':'sếp_quản_lý_dự_án','hr':'nhân_sự',
                      'environment':'môi_trường','developer':'lập_trinh_viên','department':'phòng_ban','projects':'dự_án','career':'nghề_nghiệp',
                      'opportunities':'cơ_hội','opportunity':'cơ_hội','pantry':'bữa_ăn_nhẹ','build':'xây_dựng','except':'ngoại_trừ',
                      'situation':'tình_huống','critical':'quan_trọng','care':'quan_tâm','policy':'chính_sách',
                      'interviewers':'người_phỏng_vấn','interviewer':'người_phỏng_vấn','project':'dự_án','training':'đào_tạo','management':'quản_lý'}
english_dict.update(new_eng_to_vn_dict)

for x in ['công_ty','doanh_nghiệp','cỏ','khô','tùy','hầu_như','làm_việc','công_việc','đội','đi','làm_thêm_giờ','và','lot','có','ty','công']:
  stopwords_lst.append(x)
stopwords_lst.remove('ít_khi')



# 1.3 Cac ham xu ly du lieu tho
# 1.3.1 ham process_text dung de tien xu ly van ban
def process_text(text, emoji_dict, teen_dict):
    document = text.lower() # chuyen sang chu thuong
    document = document.replace("’",'') # bo nhay don
    document = document.replace(",",' ') # bo dau phay
    document = document.replace("\n",' ') # bo xuong dong
    document = regex.sub(r'\.+', ".", document).strip() # thay the cac dau cham thanh 1 dau cham
    new_sentence =''
    for sentence in sent_tokenize(document):
        # CONVERT EMOJICON sang word
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        # CONVERT TEENCODE sang full word
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        # DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # DEL wrong words (xoa tu sai ve cu phap)
        # sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    # DEL excess blank space (xoa, chi de lai 1 khoang trang)
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

#1.3.2 Ham phan tach tieng anh, tieng viet
def check_lang(text):
    lang = detect(text)
    return lang

#1.3.3 ham dich tieng anh ra  viet
def translate_text(text, english_dict):
    new_sentence =''
    for sentence in sent_tokenize(text):
        sentence = ' '.join(english_dict[word] if word in english_dict else word for word in sentence.split())
        new_sentence = new_sentence+ sentence
    text = new_sentence
    return text

#1.3.4 Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

#1.3.5 Ham xu ly cac tu phu dinh, dac biet
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if ('không' in text_lst) or ('khong' in text_lst) or ('chả' in text_lst) or ('chẳng' in text_lst) or ('ít' in text_lst) or ('it' in text_lst) or ('hiếm' in text_lst) or ('rất' in text_lst) or ('quá' in text_lst) or ('có' in text_lst) or ('khá' in text_lst) or ('hơi' in text_lst):
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  (word == 'không') or (word == 'khong') or (word == 'chả') or (word == 'chẳng') or (word == 'ít') or (word == 'it') or (word == 'hiếm') or (word == 'rất') or (word == 'quá') or (word == 'có') or (word == 'khá') or (word == 'hơi'):
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

#1.3.6 ham tao tu ghep tieng viet theo loai tu
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.',' ')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

#1.3.7 ham loai bo tu tieng viet thuoc stop list
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# 1.4. # hàm nhap lieu comment moi
def text_valid():
  text = input('Nhap noi dung: ')
  while True:
    if text =='' or text.isnumeric() or text.isspace():
      print('noi dung nhap khong hop le, vui long nhap lai')
      text = input('Nhap noi dung: ')
    else:
      break
  return text

# 1.5. hàm đếm từ positive, negative
def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []
    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)
    return word_count, word_list

# 1.6. Hàm truy vấn công ty
def find_id(id,data,col_id):
  detail = data[data[col_id] == int(id)]
  return detail
def find_name(name,data,col_name):
  detail = data[data[col_name] == str(name)]
  return detail

#1.7. hàm lấy danh sach id công ty có info/review tương tự nhau
# function cần thiết
def get_recommendations(df, id, cosine_sim, nums=5):
    # Get the index of the company that matches the id
    matching_indices = df.index[df['id'] == id].tolist()
    if not matching_indices:
        print(f"No company found with ID: {id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]
    # Get the pairwise similarity scores of all companies with that company
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the companies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the nums most similar companies (Ignoring the company itself)
    sim_scores = sim_scores[1:nums+1]
    # Get the company indices
    company_indices = [i[0] for i in sim_scores]

    # Return the top n most similar companies as a DataFrame
    return df.iloc[company_indices]

# Hiển thị đề xuất ra bảng
def display_recommended_companies(recommended_companies, cols=5):
    for i in range(0, len(recommended_companies), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_companies):
                company = recommended_companies.iloc[i + j]
                with col:                       
                    st.write(company['company_name'])                  
                    expander = st.expander(f"Company overview")
                    company_description = company['company_overview']
                    truncated_description = ' '.join(company_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")        

# 1.8. Hàm lấy các công ty có thông tin/reviews tương tự với comment
# Hàm tạo kết quả dạng dictionary
def get_result_com(data,col_name):
  results_com = {}
  for idx, row in data[[col_name]].iterrows():
      similar_indices = com_cosine_similar[idx].argsort()
      similar_items = [(com_cosine_similar[idx][i]) for i in similar_indices]
      # similar_items = [(com_cosine_similar[idx][i], data[[col_name]].index[i]) for i in similar_indices]
      # print(similar_items)
      results_com[idx] = similar_items
  results_com = dict(sorted(results_com.items(), key=lambda item: item[1], reverse=True))
  return results_com
# hàm lay toi da k id, value lon nhat
def get_id_com(result,data,col_id,k=1):
  key_list = [i for i in result.keys()]

  idx_list = []
  rate_list = []
  for i in range(k):
    id = data.loc[key_list[i],col_id]
    idx_list.append(int(id))
    rate_list.append(float(result[key_list[i]][0]))
  return idx_list


# 1.9 ham ve wordcloud
# Hàm vẽ biểu đồ wordcloud cho tất cả công ty
def ghep_all(data,col_text):
    # Ghép toàn bộ văn bản trong cột thành một chuỗi
    text = " ".join(data[col_text].dropna().astype(str).tolist())
    return text




# 2. ĐỌC MODEL
# 2.1. cac model dung cho goi y cac cong ty tuong tu nhau
import pickle
with open('01_cosin_info.pkl', 'rb') as file:  
    cosin_info = pickle.load(file)
with open('01_vector_info.pkl', 'rb') as file:  
    vector_info = pickle.load(file)

# 2.2. các model gợi ý các công ty có review tương tự nhau
with open('02_cosin_reviews.pkl', 'rb') as file:  
    cosin_reviews = pickle.load(file)
with open('02_vector_reviews.pkl', 'rb') as file:  
    vector_reviews = pickle.load(file)




# 3. GUI
line1 = st.sidebar.title('FINAL PROJECT')
menu = ["Xác định vấn đề", "01 - Content Based Similarity", "02 - Recommend Classification"]
choice = st.sidebar.selectbox('Menu', menu)
line1 = st.sidebar.write('\n')
line2 = st.sidebar.write('\n')
line3 = st.sidebar.write('\n')
line4 = st.sidebar.write('\n')
line5 = st.sidebar.write('\n')
line6 = st.sidebar.write('\n')
line6 = st.sidebar.write('\n')
line7 = st.sidebar.write('\n')
line8 = st.sidebar.write('\n')
info1 = st.sidebar.write('Thông tin học viên:')
info2 = st.sidebar.write(''' Đặng Thanh Dung \n dungdang0427@gmail.com \n DL07_K304''')


if choice == 'Xác định vấn đề':  
    st.title('Đồ án tốt nghiệp Data Science - Machine Learning')  
    st.subheader("Xác định vấn đề")
    st.write('Trong Project có 2 vấn đề liên quan gồm:')
    st.write('#### 1. Gợi ý các công ty có thông tin tương tự nhau:')
    st.write("""
- Dữ liệu gốc: các thông tin mô tả về công ty đăng tải trên ITViec.\n
- Tiền xử lý dữ liệu: áp dụng các thư viện, công cụ phù hợp để xử lý ngôn ngữ (tiếng Anh, tiếng Việt).\n
    + Chuyển đổi về kiểu chữ thường, loại bỏ ký tự đặc biệt, thay thế emojicon, teancode, khoảng trắng
    + Thêm cột để phân biệt loại ngôn ngữ Anh, Việt của các thông tin mô tả
    + Dịch tiếng Anh sang tiếng Việt
    + Chuẩn hóa tiếng việt, xử lý các từ đặc biệt, ghép các từ thành từ ghép có nghĩa
    + Xóa các từ thuộc stopwords tiếng Việt
- Dữ liệu đầu vào: là các thông tin mô tả công ty đã được tiền xử lý dữ liệu và có ý nghĩa cả về ngôn ngữ và máy học. Từ đó, tạo ra các đặc trưng (features) làm đầu vào cho các mô hình Machine learning.\n
- Thuật toán: sử dụng Gensim, Cosin, Linear Kernel để tiến hành xác định các công ty có nội dung mô tả tương tự nhau.\n
- Kết quả: \n 
    + Lựa chọn mô hình phù hợp với dữ liệu, xác định được các công ty có thông tin mô tả gần giống nhất. \n 
    + Khi nhập nội dung tìm kiếm mới, sẽ hiển thị được các công ty có thông tin mô tả gần giống nhất. \n
    """)
    st.image('anh_slide/content_based.png',width=300,caption='')  

    st.write("#### 2. Dự đoán khả năng 'recommend or not' công ty:")
    st.write("""
- Dữ liệu gốc: các đánh giá (reviews) của các ứng viên/nhân viên trên trang ITViec.\n
- Tiền xử lý dữ liệu: áp dụng các thư viện, công cụ phù hợp để xử lý ngôn ngữ (tiếng Anh, tiếng Việt).\n
    + Chuyển đổi về kiểu chữ thường, loại bỏ ký tự đặc biệt, thay thế emojicon, teancode, khoảng trắng
    + Thêm cột để phân biệt loại ngôn ngữ Anh, Việt của các reviews
    + Dịch tiếng Anh sang tiếng Việt
    + Chuẩn hóa tiếng việt, xử lý các từ đặc biệt, ghép các từ thành từ ghép có nghĩa
    + Xóa các từ thuộc stopwords tiếng Việt
- Dữ liệu đầu vào: là các reviews ứng với mỗi công ty đã được tiền xử lý dữ liệu và có ý nghĩa cả về ngôn ngữ và máy học. Từ đó, tạo ra các đặc trưng (features) làm đầu vào cho các mô hình Machine learning.\n
- Thuật toán: 
    + Sklearn: sử dụng Navie Bayes, Logistic regression, KNN, Decision tree, Random forest, Ada Boost để tiến hành phân tích.\n
    + Spark: sử dụng Navie Bayes, Logistic regression, Random forest để tiến hành phân tích.\n
    + Riêng cosin similarity được sử dụng khi xác định các công ty có reviews tương tự với nhau và với nội dung tìm kiếm.
- Kết quả: \n 
    + Lựa chọn mô hình phù hợp với dữ liệu, có thể dự đoán việc "recommend or not" công ty.
    + Khi chọn các tiêu chí cụ thể sẽ hiển thị các công ty được đề xuất và thỏa mãn các tiêu chí này.\n 
    + Khi chọn từng công ty sẽ hiển thị công ty có được đề xuất hay không? Nếu có sẽ hiển thị chi tiết về công ty và các công ty có reviews gần giống nhất. \n 
    + Khi nhập nội dung tìm kiếm mới, sẽ hiển thị các công ty được đề xuất và có reviews gần giống nhất. \n
    """)
    st.image('anh_slide/recommend.png',width=300,caption='')   



elif choice == '01 - Content Based Similarity':
    st.subheader("01 - Content Based Similarity")

    st.write("#### 1. Tổng quan")
    df_info = pd.read_csv('03_gui_company.csv')

    st.write('##### Companies')
    st.write('Căn cứ thông tin mô tả các công ty trên trang ITViec với bức tranh tổng quan như sau:')
    plt.figure(figsize=(12,3))
    plt.subplot(1,2,1)
    plt.title('Loại công ty')
    sns.countplot(data = df_info,x='company_type',hue='company_type',legend=False)
    plt.xticks(rotation=90)
    plt.subplot(1,2,2)
    plt.title('Quy mô')
    sns.countplot(data = df_info,x='company_size',hue='company_size',legend=False)
    plt.xticks(rotation=90)
    st.pyplot(plt)    
    st.write('-----------------------------------------------------')
    plt.figure(figsize=(12,3))   
    plt.subplot(1,2,1)
    plt.title('Thời gian làm việc')
    sns.countplot(data = df_info,x='working_days',hue='working_days',legend=False)
    plt.xticks(rotation=90)
    plt.subplot(1,2,2)
    plt.title('Chính sách làm thêm giờ')
    sns.countplot(data = df_info,x='overtime_policy',hue='overtime_policy',legend=False)
    plt.xticks(rotation=90)
    st.pyplot(plt)    
    st.write('-----------------------------------------------------')
    plt.figure(figsize=(12,3))
    plt.title('Quốc gia')
    sns.countplot(data = df_info,x='country',hue='country')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.write('-----------------------------------------------------')
    plt.figure(figsize=(12,3))
    plt.title('Ngành nghề')
    sns.countplot(data = df_info,x='company_industry',hue='company_industry',legend=False)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.write('-----------------------------------------------------')
    st.write('Nhận xét: Các công ty chủ yếu thuộc lĩnh vực IT, có quy mô vừa và nhỏ, thời gian làm việc từ thứ 2 đến thứ 6, không làm thêm giờ, có trụ sở tại Việt Nam')
    st.write('Bài toán xác định các công ty có thông tin mô tả tương tự nhau sẽ được áp dụng cho 478 công ty.')

    st.write("#### 2. Biến input")
    st.write('''Căn cứ các thông tin về 'Company overview (company_overview), Our key skills (key_skills), Why you'll love working here (why_working)'. Tác giả thực hiện các bước biến đổi như sau:\n
- Thực hiện gộp tất cả thông tin mô tả trên theo công ty để tạo thành biến 'company_info'
- Dịch toàn bộ nội dung sang tiếng Việt
- Tiền xử lý dữ liệu tiếng Việt''')
    

    st.write("#### 3. Xây dựng mô hình và đánh giá")
    st.write("""Tác giả sử dụng 03 mô hình gồm: Gensim, Cosin, Linear Kernel.""")
    st.write('Chi tiết cụ thể như sau:')
    st.write('-------------------------------------------------------------')
    st.write('##### Gensim')
    st.write("""Các bước thực hiện gồm: \n
- Vector hóa bằng TfidfVectorizer \n
- Tạo ma trận similarity \n
- Tạo list chứa kết quả so sánh sự tương quan giữa công ty được chọn và các công ty khác \n
- Tạo hàm lấy ra 3 công ty tương quan nhất \n
            """)
    st.write("Ví dụ: Xác định 3 công ty có thông tin tương tự với công ty có id=0")
    st.image('anh_slide/cont_gensim.png',width=500)

    st.write('-------------------------------------------------------------')    
    st.write('##### Cosin')
    st.write("""Các bước thực hiện gồm: \n
- Vector hóa bằng TfidfVectorizer \n
- Tạo ma trận similarity \n
- Tạo dictionary chứa kết quả so sánh sự tương quan giữa các công ty (chọn 3 công ty có tương quan nhất) \n
- Tạo hàm lấy ra id các công ty tương quan \n
            """)
    st.write("Ví dụ: Xác định 3 công ty có thông tin tương tự với công ty có id=0")
    st.image('anh_slide/cont_cosin.png',width=500)

    st.write('-------------------------------------------------------------')    
    st.write('##### Linear Kernel')
    st.write("""Các bước thực hiện gồm: \n
- Vector hóa bằng TfidfVectorizer \n
- Tạo ma trận similarity \n
- Tạo dictionary chứa kết quả so sánh sự tương quan giữa các công ty (chọn 3 công ty có tương quan nhất) \n
- Tạo hàm lấy ra id các công ty tương quan \n
            """)
    st.write("Ví dụ: Xác định 3 công ty có thông tin tương tự với công ty có id=0")
    st.image('anh_slide/cont_linear.png',width=500)
    st.write('-------------------------------------------------------------')  
    st.write("""Nhận xét: Từ kết quả trên, tác giả lựa chọn Cosin làm mô hình để xác định các công ty có thông tin tương tự nhau vì kết quả có độ chính xác cao và dễ sử dụng.""")

    st.write("#### 4. Hiển thị công ty có thông tin mô tả tương tự nhau")
    # Tạo ma trận tương tự
    info_cosin_similar = cosine_similarity(cosin_info,cosin_info)

    # Đọc dữ liệu sản phẩm
    if 'random_companies' not in st.session_state:
        df_companies = df_info
        st.session_state.random_companies = df_companies.sample(n=10, random_state=42)
        # st.session_state.random_companies = df_companies
    else:
        df_companies = df_info 

    # Kiểm tra xem 'selected_id' đã có trong session_state hay chưa
    if 'selected_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_id = None

    # Theo cách cho người dùng chọn công ty từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    company_options = [(row['company_name'], row['id']) for index, row in st.session_state.random_companies.iterrows()]
    # st.session_state.random_companies
    # Tạo một dropdown với options là các tuple này
    selected_company = st.selectbox(
        "Chọn công ty",
        options=company_options,
        format_func=lambda x: x[0]  # Hiển thị tên công ty
    )
    # Display the selected company
    st.write("Bạn đã chọn:", selected_company)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_id = selected_company[1]

    if st.session_state.selected_id:
        # st.write("id: ", st.session_state.selected_id)
        # Hiển thị thông tin sản phẩm được chọn
        selected_company = df_info[df_info['id'] == st.session_state.selected_id]

        if not selected_company.empty:
            # st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_company['company_name'].values[0])

            company_description = selected_company['company_overview'].values[0]
            truncated_description = ' '.join(company_description.split()[:100])
            st.write('##### Giới thiệu:')
            st.write(truncated_description, '...')

            st.write('##### Các công ty liên quan:')
            recommendations = get_recommendations(df_info, st.session_state.selected_id, cosine_sim=info_cosin_similar, nums=3) 
            display_recommended_companies(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy công ty với ID: {st.session_state.selected_id}")


    st.write("#### 5. Hiển thị công ty có thông tin mô tả tương tự nội dung tìm kiếm")
    type = st.checkbox("Nhập tìm kiếm")
    if type:        
        comment = st.text_area(label="Nhập nội dung:")
    submit = st.button("Submit")
    if submit:  
        comment = process_text(str(comment), emoji_dict, teen_dict)
        if check_lang(str(comment)) == 'vi':
            comment = translate_text(str(comment),english_dict)
        else:
            translation = translator.translate(comment, src=check_lang(str(comment)), dest='vi')
            comment = translation.text
        comment = covert_unicode(str(comment))
        comment = process_postag_thesea(str(comment))
        comment = remove_stopword(str(comment), stopwords_lst) 
        # st.write("Nôi dung đã tiền xử lý tiếng việt:")
        # st.write(comment)
        com_transformed = vector_info.transform([comment])
        com_cosine_similar = cosine_similarity(cosin_info,com_transformed)
        st.write('##### Các công ty liên quan:')
        results_com = get_result_com(df_info,'company_info')
        idx_list_com = get_id_com(results_com,df_info,'id',k=3)
        st.dataframe(df_info.drop(['review_new','company_info','prediction'],axis=1).loc[idx_list_com,:].T)




elif choice == '02 - Recommend Classification':
    st.subheader("02 - Recommend Classification")

    st.write("#### 1. Tổng quan")

    df_info = pd.read_csv('03_gui_company.csv')
    df_class = df_info.dropna(subset='prediction').reset_index()
    df_recommend = df_info[df_info['prediction']==1].reset_index()

    st.write('##### Reviews')
    df_reviews = pd.read_csv('03_gui_review.csv') 

    plt.figure(figsize=(6,3))
    sns.countplot(x='Rating', data=df_reviews, hue='Rating')
    plt.title('Thống kê số lượng reviews ở mỗi điểm rating')
    st.pyplot(plt)
    st.write('Nhận xét: Đa số reviews có điểm rating >=4, số lượng reviews có điểm rating <3 chiếm khá ít.')
    st.write('-----------------------------------------------------')  
    #Tạo wordcloud
    text_all = ghep_all(df_info,'review_new')
    wordcloud = WordCloud(
        # width=600,
        # height=300,
        background_color="white",max_words=40
    ).generate(text_all)
    # Hiển thị WordCloud
    plt.figure(figsize=(3, 3))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title('Biểu đồ wordclouds theo reviews của các công ty', fontsize=9)
    st.pyplot(plt)
    st.write('Nhận xét: Đa số các từ trong reviews có nội dung tích cực, thể hiện các tiêu chí quan tâm của ứng viên/nhân viên về công ty như: lương, môi trường làm việc, đồng nghiệp, sếp, dự án,...')
    st.write('-----------------------------------------------------')  
    describe_type = st.radio("## Chọn loại thống kê mô tả:", ("Tất cả", "Theo công ty"))
    if describe_type =='Tất cả':
        st.write(df_reviews[['Rating','Salary & benefits','Training & learning','Management cares about me','Culture & fun','Office & workspace']].describe())
    else:
        st.write(df_reviews.groupby('Company Name')[['Rating','Salary & benefits','Training & learning','Management cares about me','Culture & fun','Office & workspace']].agg(
                    avg_rating = pd.NamedAgg(column='Rating',aggfunc='mean'),
                    avg_salary_benefit = pd.NamedAgg(column='Salary & benefits',aggfunc='mean'),
                    avg_traing_learning = pd.NamedAgg(column='Training & learning',aggfunc='mean'),
                    avg_management_care = pd.NamedAgg(column='Management cares about me',aggfunc='mean'),
                    avg_culture_fun = pd.NamedAgg(column='Culture & fun',aggfunc='mean'),
                    avg_office_workspace = pd.NamedAgg(column='Office & workspace',aggfunc='mean')).sort_values(by='Company Name',ascending=True))
    st.write('-----------------------------------------------------')  
    plt.figure(figsize=(12,6))    
    sns.countplot(data = df_reviews,x='Recommend',hue='Rating')
    plt.title('Thống kê số lượng reviews ở mỗi điểm rating của trường hợp đề xuất (Yes) hoặc không đề xuất (No)')
    st.pyplot(plt)  
    st.write("""Nhận xét: Có thể thấy trong dữ liệu đầu vào, một số reviews có sự mâu thuẫn giữa việc đề xuất giới thiệu công ty và điểm rating ứng viên đánh giá.""")
    st.write('-----------------------------------------------------')
    st.image('anh_slide/review_view.png',width=500)
    st.write('Nhận xét: Bài toán classification sẽ được áp dụng cho 180 công ty có reviews.')

    st.write("#### 2. Biến input và Biến output")
    st.write("""Căn cứ các thông tin về 'What i liked, Suggestions for improvement'. Tác giả thực hiện các bước biến đổi như sau:\n
- Thực hiện gộp tất cả nội dung trên theo từng reviews
- Dịch toàn bộ nội dung sang tiếng Việt
- Tiền xử lý dữ liệu tiếng Việt
- Gộp tất cả các reviews theo từng công ty tạo thành biến input 'review_new' """)
    st.write('''Biến output được quy đổi từ giá trị recommend trung bình của mỗi công ty như sau: \n
- Tỷ lệ recommend trung bình <= 0.8 ==> biến output 'recommend_new' = 0
- Tỷ lệ recommend trung bình > 0.8 ==> biến output 'recommend_new' = 1''')
    st.image('anh_slide/class_output.png',width=300)
    st.write("Nhận xét: Biến output có sự mất cân bằng về giá trị, do đó tác giả sử dụng tham số class_weight='balanced' khi áp dụng mô hình. ")

    st.write("#### 3. Xây dựng mô hình và đánh giá")
    st.write('#### Sklearn:')
    st.write('Skearn với 06 mô hình gồm: Navie Bayes, KNN, Logistic regression, Decision tree, Random forest, Ada boost.\n Để đánh giá hiệu quả của 06 mô hình trên, tác giả đã thực hiện đo lường thời gian và sử dụng cross validation tính điểm accuracy trung bình.\n Kết quả như các ảnh sau:''')
    st.image("anh_slide/class_1.png")
    st.image("anh_slide/class_2.png")
    st.write('Chi tiết cụ thể như sau:')
    st.write('-------------------------------------------------------------')
    st.write('##### Naive Bayes')
    st.image('anh_slide/class_nb.png',width=500)
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.6666666666666666
            - f1_score: 0.7090909090909091
            - precision_score: 0.84375
            - recall_score: 0.6666666666666666
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Logistic regression')
    st.image('anh_slide/class_lg.png',width=500)
    st.write("""Các chỉ số đánh giá mô hình:\n
        - accuracy_score: 0.7777777777777778
        - f1_score: 0.7976190476190476
        - precision_score: 0.8358974358974359
        - recall_score: 0.7777777777777778
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### KNN')
    st.image('anh_slide/class_knn.png',width=500)
    st.write("""Các chỉ số đánh giá mô hình:\n
        - accuracy_score: 0.7777777777777778
        - f1_score: 0.7291666666666666
        - precision_score: 0.6862745098039215
        - recall_score: 0.7777777777777778
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Decision tree')
    st.image('anh_slide/class_dt.png',width=500)
    st.write("""Các chỉ số đánh giá mô hình:\n
        - accuracy_score: 0.75
        - f1_score: 0.7684210526315789
        - precision_score: 0.7962962962962963
        - recall_score: 0.75
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Random forest')
    st.image('anh_slide/class_rf.png',width=500)
    st.write("""Các chỉ số đánh giá mô hình:\n
        - accuracy_score: 0.8611111111111112
        - f1_score: 0.8412698412698413
        - precision_score: 0.8434343434343434
        - recall_score: 0.8611111111111112
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Ada Boost')
    st.image('anh_slide/class_ab.png',width=500)
    st.write("""Các chỉ số đánh giá mô hình:\n
        - accuracy_score: 0.75
        - f1_score: 0.757496740547588
        - precision_score: 0.7660098522167487
        - recall_score: 0.75
            """)
    st.write('-------------------------------------------------------------')  
    st.write('#### Spark:')
    st.write('Spark với 03 mô hình gồm: Navie Bayes, Logistic regression, Random forest')
    st.write('Chi tiết cụ thể như sau:')
    st.write('-------------------------------------------------------------')
    st.write('##### Naive Bayes')
    st.image('anh_slide/spark_nb.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
        - ACC: 0.675
        - AUC: 0.36274509803921573
            """)
    st.write('-------------------------------------------------------------')  
    st.write('##### Logistic regression')
    st.image('anh_slide/spark_lg.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
        - ACC: 0.725
        - AUC: 0.5833333333333333
        """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Random forest')
    st.image('anh_slide/spark_rf.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
        - ACC: 0.825
        - AUC: 0.7058823529411765
            """)
    st.write('-------------------------------------------------------------')    
    st.write("""Nhận xét: Từ kết quả trên, tác giả lựa chọn Random forest của Sklearn làm mô hình để dự đoán 'recommend or not' công ty vì có điểm accuracy cao và thời gian thực hiện ngắn.""")

    st.write("#### 4. Hiển thị công ty")
    recommend_type = st.radio("## Tùy chọn:", ("Theo các tiêu chí","Theo công ty"))
    if recommend_type =='Theo các tiêu chí':
        salary_benefits = st.slider("Salary & benefits", 1, 5, value=4,key=1)
        training_learning = st.slider("Training & learing", 1, 5, value=4,key=2)
        management_cares_me = st.slider("Management cares me", 1, 5, value=4,key=3)
        culture_fun = st.slider("Culture & fun", 1, 5, value=4,key=4)
        office_workspace = st.slider("Office & workspace", 1, 5, value=4,key=5)
        recommend_id_list = df_class.loc[(df_class['salary_benefits'] >= salary_benefits) & (df_class['training_learning'] >= training_learning) & (df_class['management_cares_me'] >= management_cares_me) & (df_class['culture_fun'] >= culture_fun) & (df_class['office_workspace'] >= office_workspace) & (df_class['prediction']==1),'id'].to_list()
        recommend_name_list = df_class.loc[(df_class['salary_benefits'] >= salary_benefits) & (df_class['training_learning'] >= training_learning) & (df_class['management_cares_me'] >= management_cares_me) & (df_class['culture_fun'] >= culture_fun) & (df_class['office_workspace'] >= office_workspace) & (df_class['prediction']==1),'company_name'].to_list()
        if len(recommend_id_list) !=0:
            st.write('==> Có '+str(len(recommend_id_list))+' công ty được đề xuất là: ')
            st.write('Thông tin về các công ty:')
            company_detail = pd.DataFrame()
            for id in recommend_id_list:              
                detail = find_id(id,df_info.drop(['review_new','prediction','company_info'],axis=1),'id')
                company_detail = pd.concat([company_detail,detail],axis=0)
            st.write(company_detail)
        else:
            st.write('==> Không có công ty nào được đề xuất')
    else:
        # Xác định ma trận tương tự
        review_cosine_similar = cosine_similarity(cosin_reviews,cosin_reviews)

        # Đọc dữ liệu sản phẩm
        if 'random_companies2' not in st.session_state:
            df_companies2 = df_info
            st.session_state.random_companies2 = df_companies2
        else:
            df_companies2 = df_info 

        # Kiểm tra xem 'selected_id' đã có trong session_state hay chưa
        if 'selected_id' not in st.session_state:
            # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
            st.session_state.selected_id = None

        # Theo cách cho người dùng chọn công ty từ dropdown
        # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        company_options = [(row['company_name'], row['id']) for index, row in st.session_state.random_companies2.iterrows()]
        # st.session_state.random_companies2
        # Tạo một dropdown với options là các tuple này
        selected_company = st.selectbox(
            "Chọn công ty",
            options=company_options,
            format_func=lambda x: x[0]  # Hiển thị tên công ty
        )
        # Display the selected company
        st.write("Bạn đã chọn:", selected_company)

        # Tạo các list chứa id và tên của công ty được đề xuất
        recommended_companies = df_recommend['company_name'].to_list()

        if selected_company[0] in recommended_companies:
            # Cập nhật session_state dựa trên lựa chọn hiện tại
            st.session_state.selected_id = selected_company[1]

            if st.session_state.selected_id:
                # st.write("id: ", st.session_state.selected_id)
                # Hiển thị thông tin sản phẩm được chọn
                selected_company = df_class[df_class['id'] == st.session_state.selected_id]

                if not selected_company.empty:
                    # st.write('#### Bạn vừa chọn:')
                    st.write('### ', selected_company['company_name'].values[0])

                    company_description = selected_company['company_overview'].values[0]
                    truncated_description = ' '.join(company_description.split()[:100])
                    st.write('##### Giới thiệu:')
                    st.write(truncated_description, '...')

                    st.write('##### Chi tiết về công ty:')                
                    company_detail = find_name(selected_company['company_name'].values[0],df_info.drop(['company_overview','review_new','prediction','company_info'],axis=1),'company_name').T
                    st.write(company_detail)

                    st.write('##### Các công ty liên quan:')
                    recommendations = get_recommendations(df_recommend, st.session_state.selected_id, cosine_sim=review_cosine_similar, nums=3) 
                    display_recommended_companies(recommendations, cols=3)
                else:
                    st.write(f"Không tìm thấy công ty với ID: {st.session_state.selected_id}")
        else:
            st.write("==> Không tìm thấy dữ liệu thỏa mãn do công ty không được đề xuất.")


    st.write("#### 5. Hiển thị công ty được đề xuất và có reviews tương tự nội dung tìm kiếm")
    type2 = st.checkbox("Nhập tìm kiếm")
    if type2:        
        comment2 = st.text_area(label="Nhập nội dung:")
    submit2 = st.button("Submit")
    if submit2:    
        comment2 = process_text(str(comment2), emoji_dict, teen_dict)
        if check_lang(str(comment2)) == 'vi':
            comment2 = translate_text(str(comment2),english_dict)
        else:
            translation2 = translator.translate(comment2, src=check_lang(str(comment2)), dest='vi')
            comment2 = translation2.text
        comment2 = covert_unicode(str(comment2))
        comment2 = process_postag_thesea(str(comment2))
        comment2 = remove_stopword(str(comment2), stopwords_lst) 
        # st.write("Nôi dung đã tiền xử lý tiếng việt:")
        # st.write(comment)
        com_transformed2 = vector_reviews.transform([comment2])
        com_cosine_similar = cosine_similarity(cosin_reviews,com_transformed2)
        st.write('##### Các công ty liên quan:')
        results_com2 = get_result_com(df_recommend,'review_new')
        idx_list_com2 = get_id_com(results_com2,df_recommend,'id',k=3)
        st.dataframe(df_info.drop(['review_new','prediction','company_info'],axis=1).loc[idx_list_com2,:].T)

