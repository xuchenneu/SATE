import argparse
import re


def read_file(input_path):
    sentences = []
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            sen_temp = line.strip()
            sentences.append(sen_temp)
    return sentences


def write_file(output_path, sentences):
    with open(output_path, 'w', encoding='utf8') as f:
        for line in sentences:
            f.write(line + '\n')
    return


def remove_tag(sentences):
    ### 去掉(Applaus)等带括号的tag ###
    sen_new = []
    for line in sentences:
        sen_temp = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", line)
        sen_new.append(sen_temp)
    return sen_new


def remove_beginning_punctuation(sentences):
    #### 去掉开头的逗号等符号 ###
    remove_punctuation = [',', '.', '?', ':', '-', ' ']
    sen_new = []
    for line in sentences:
        if len(line) > 0 and line[0] in remove_punctuation:
            sen_temp = line
            for p in remove_punctuation:
                sen_temp = sen_temp.lstrip(p)
            sen_new.append(sen_temp)
        else:
            sen_new.append(line)
    return sen_new


def remove_ending_punctuation(sentences):
    ### 去掉句尾的逗号等符号 ###
    remove_punctuation = [',', ':', ' ']
    sen_new = []
    for line in sentences:
        # print(line[-1])
        if len(line) > 0 and line[-1] in remove_punctuation:
            sen_temp = line
            for p in remove_punctuation:
                sen_temp = sen_temp.rstrip(p)
            sen_new.append(sen_temp)
        else:
            sen_new.append(line)
    return sen_new


def remove_space(sentences):
    ### 去掉首尾的空格，以及连续的空格 ###
    sen_new = []
    for line in sentences:
        sen_temp = line.strip()
        sen_temp = ' '.join(sen_temp.split())
        sen_new.append(sen_temp)
    return sen_new


def remove_special_tag(sentences):
    ### 去掉双破折号 -- ，可选 ###
    sen_new = []
    for line in sentences:
        sen_temp = line.replace('--', '—')
        sen_new.append(sen_temp)
    return sen_new


def first_letter_upper(sentences):
    ### 将首字母大写 ###
    sen_new = []
    for line in sentences:
        if len(line) > 0 and line[0].isalpha() and line[0].islower():
            l = list(line)
            l[0] = l[0].upper()
            sen_temp = ''.join(l)
            sen_new.append(sen_temp)
            continue
        else:
            sen_new.append(line)
    return sen_new


def add_last_punctuation(sentences):
    ### 给末尾没有标点的句子加句号 . ###
    sen_new = []
    for line in sentences:
        if len(line) > 0 and line[-1].isalpha():
            sen_temp = line + '.'
            sen_new.append(sen_temp)
        else:
            sen_new.append(line)
    return sen_new


def process(args):
    input_path = args.input_absolute_path
    output_path = args.output_absolute_path
    sentences = read_file(input_path)
    sentences = remove_tag(sentences)
    # sentences = remove_beginning_punctuation(sentences)
    # sentences = remove_ending_punctuation(sentences)
    sentences = remove_special_tag(sentences)
    # sentences = remove_space(sentences)
    # sentences = first_letter_upper(sentences)
    # sentences = add_last_punctuation(sentences)
    write_file(output_path, sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_absolute_path", required=True, type=str)  # 输入文件绝对路径
    parser.add_argument("--output_absolute_path", required=True, type=str)  # 输出文件绝对路径
    args = parser.parse_args()

    process(args)


if __name__ == '__main__':
    main()
