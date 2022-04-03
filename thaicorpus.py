import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.util import *
from pythainlp.corpus.common import thai_stopwords


def padding_text_decoder(sentence, maxseq):
    dataset = []
    n = maxseq - len(sentence)
    if n > 0:
        dataset.append(sentence + (["<EOS>"] * n))
    elif n < 0:
        dataset.append(sentence[0:maxseq])
    else:
        dataset.append(sentence)
    return dataset


if __name__ == '__main__':
    text = "มาตรา 2 "
    text = " ".join(text.split())
    # out1 = word_tokenize(text, engine='dict')
    # out2 = word_tokenize(text, engine='mm')
    # out4 = word_tokenize(text, engine='pylexto')
    word_token = word_tokenize(text, engine='newmm')

    print(word_token)
    # print(out1)
    # print(out2)
    # print(out4)
    # print(thai_stopwords())
    # print(len(thai_stopwords()))
    # print(pythainlp.thai_characters)
    # text_encoder = list((set(word_token) - set(thai_stopwords())) - set({'?', '!', ' '})) #1
    text_encoder = list(set(word_token) - set({'?', '!'}))  # 2

    text_decoder = padding_text_decoder(text_encoder,10)

    word_token2 = []

    for text_string in list(word_token):
        if not isthai(text_string):
            for text in list(text_string):
                word_token2.append(digit_to_text(text))
        else:
            word_token2.append(text_string)
    print(word_token2)



