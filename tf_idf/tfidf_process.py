import os
import jieba
import math

jieba.load_userdict("dict.txt")

base_path = os.getcwd().replace('\\', '/') + '/../'
train_data_path = base_path + 'data/train'
terms_data_path = base_path + 'data/terms/all_files_terms.txt'
output_result_path = base_path + 'output_result.txt'
filter_terms = ['\n', '\t', '　', '　　', ' ', '（', '）', '：', '。', '，', '.', '；', '、', '“', '”']
delete_terms = [' ']
stopwords = {}
stop_word_path = 'stop_words.txt'
encoding = 'utf-8'


def cut_file():
    file_list = os.listdir(train_data_path)
    file_terms = open(terms_data_path, 'w+', encoding=encoding)
    i = 0
    for file in file_list:
        i += 1
        _file = open(train_data_path + '/' + file, "r", encoding=encoding)
        content = terms_filter(''.join(_file.readlines()))
        cut_content_list = cut_words(content)
        cut_content = ''
        for word in cut_content_list:
            if word not in stopwords:
                cut_content += word
                cut_content += ' '
        file_terms.write(file + ':')
        file_terms.write(cut_content[0:-1])
        file_terms.write('\n')


def load_stop_word():
    f = open(stop_word_path, 'r', encoding=encoding)
    for word in f:
        w = word.encode('utf-8').decode('utf-8-sig').strip()
        stopwords[w] = w
    f.close()


def cut_words(content):
    cut_content = jieba.cut(content, cut_all=False)
    return cut_content


def terms_filter(content):
    for word in filter_terms:
        content = content.replace(word, "")
    return content


def cal_tf_idf():
    file_counter = 0
    terms_per_file = {}
    terms_all_files = {}
    with open(terms_data_path, 'r', encoding=encoding) as file:
        for line in file:
            name_terms = line.strip("\n").split(":")
            name = name_terms[0]
            terms = name_terms[1].split(' ')
            file_counter += 1
            per_file_term_dict = {}
            for term in terms:
                if term not in terms_all_files:
                    terms_all_files[term] = 1
                    per_file_term_dict[term] = 1
                else:
                    if term not in per_file_term_dict:
                        terms_all_files[term] += 1
                        per_file_term_dict[term] = 1
                if name not in terms_per_file:
                    terms_per_file[name] = {}
                if term not in terms_per_file[name]:
                    terms_per_file[name][term] = []
                    terms_per_file[name][term].append(terms.count(term))
                    terms_per_file[name][term].append(len(terms))
    output_result = []
    for filename in terms_per_file.keys():
        result = {}
        result.clear()
        for term in terms_per_file[filename].keys():
            term_num = terms_per_file[filename][term][0]
            total_num = terms_per_file[filename][term][1]
            all_num = terms_all_files[term]
            result[term] = (term_num / total_num) * (math.log10((file_counter+1) / (all_num+1))+1)  #smoothing
        output_result.append(filename)
        output_result.extend(result.items())
    f = open(output_result_path, "w+", encoding=encoding)
    for o_result in output_result:
        for s in o_result:
            f.write(str(s))
        f.write("\n")
    f.close()


if __name__ == '__main__':
    load_stop_word()
    cut_file()
    cal_tf_idf()
