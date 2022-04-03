from pythainlp.tokenize import word_tokenize
from pythainlp.util import *
import numpy as np
from numpy import array
from gensim.models import Word2Vec
import difflib
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import Input

wv_model = Word2Vec.load('corpus.th.model')
word_list = wv_model.wv.index_to_key


def load_data(datafile):
    dataX = []
    dataY = []
    data = open(datafile, "r").read().lower()
    for i in data.split("\n\n"):
        a = i.split("\n")
        question = a[0]
        answer = a[1]
        dataX.append(question)
        dataY.append(answer)
    return dataX, dataY


def preparingword(listword):
    word = []
    for w in listword:
        word.append(wordcut(w))
    return word

def _wordcut(sentence):
    return word_tokenize(sentence, engine='newmm')

def wordcut(sentence):
    word_token = word_tokenize(sentence, engine='newmm')
    word_token2 =[]
    for text_string in list(word_token):
        if not isthai(text_string):
            for text in list(text_string):
                word_token2.append(digit_to_text(text))
        else:
            word_token2.append(text_string)
    return word_token2

def padding_sequence(listsentence, maxseq):
    dataset = []
    for s in listsentence:
        n = maxseq - len(s)
        if n > 0:
            dataset.append(s + (["<EOS>"] * n))
        elif n < 0:
            dataset.append(s[0:maxseq])
        else:
            dataset.append(s)
    return dataset


def word_index(listword):
    dataset = []
    for sentence in listword:
        tmp = []
        for w in sentence:
            tmp.append(word2idx(w))
        dataset.append(tmp)
    return np.array(dataset)


def word2idx(word):
    index = 0
    try:
        # index = wv_model.wv.vocab[word].index
        index = wv_model.wv.key_to_index[word]
    except:
        try:
            sim = similar_word(word)
            # index = wv_model.wv.vocab[sim].index
            index = wv_model.wv.key_to_index[sim]
        except:
            # index = wv_model.wv.vocab["<NONE>"].index
            index = wv_model.wv.key_to_index["ว่างเปล่า"]
    return index


def similar_word(word):
    sim_word = difflib.get_close_matches(word, word_list)
    try:
        return sim_word[0]
    except:
        return "ว่างเปล่า"


def embedding_model():
    # define word embedding
    vocab_list = [(k, wv_model.wv[k]) for v, k in wv_model.wv.key_to_index.items()]
    embeddings_matrix = np.zeros((len(wv_model.wv.key_to_index.items()) + 1, wv_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        embeddings_matrix[i + 1] = vocab_list[i][1]

    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                # output_dim=EMBEDDING_DIM,
                                output_dim=100,
                                weights=[embeddings_matrix],
                                trainable=False, name="Embedding")
    return embedding_layer, len(embeddings_matrix)


def ende_embedding_model(n_input, n_output, n_units):
    encoder_inputs = Input(shape=(None,), name="Encoder_input")

    encoder = LSTM(n_units, return_state=True, name='Encoder_lstm')
    Shared_Embedding, vocab_size = embedding_model()
    word_embedding_context = Shared_Embedding(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(word_embedding_context)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name="Decoder_input")
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="Decoder_lstm")
    word_embedding_answer = Shared_Embedding(decoder_inputs)
    decoder_outputs, _, _ = decoder_lstm(word_embedding_answer, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax', name="Dense_layer"))
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(n_units,), name="H_state_input")
    decoder_state_input_c = Input(shape=(n_units,), name="C_state_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model


def train():
    dataX = []
    dataY = []
    X1 = []
    X2 = []
    Y = []
    n_in = 10
    n_out = 10
    max_word = len(wv_model.wv)
    # vecsize = encoded_length
    datafile = "text/message.txt"

    dataX, dataY = load_data(datafile)
    dataX = preparingword(dataX)
    dataY = preparingword(dataY)

    for sentence in dataX:
        X1.append(sentence)
    for sentence in dataY:
        Y.append(sentence)
    for sentence in dataY:
        X2.append(['_'] + sentence[0:len(sentence) - 1])

    X1 = padding_sequence(X1, n_in)
    X2 = padding_sequence(X2, n_out)
    Y = padding_sequence(Y, n_out)

    X1 = word_index(X1)
    X2 = word_index(X2)
    Y = word_index(Y)
    Y = to_categorical(Y, num_classes=max_word + 1)

    # define model
    train, infenc, infdec = ende_embedding_model(n_in, n_out, 256)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    print(train.summary())
    # train model
    train.fit([X1, X2], Y, epochs=600)

    # saving model
    infenc.save_weights("model_enc_weight.h5")
    infenc.save("model_enc.h5")
    print("Saved model to disk")
    infdec.save_weights("model_dec_weight.h5")
    infdec.save("model_dec.h5")
    print("Saved model to disk")


def test_model():
    # define model

    n_in = 10
    n_out = 10
    encoded_length = 10
    train, infenc, infdec = ende_embedding_model(n_in, n_out, 256)
    # load weights
    infenc.load_weights("model_enc.h5")
    infdec.load_weights("model_dec.h5")

    # start prediction
    while True:

        input_data = input()
        input_data = preparingword([input_data])
        input_data = padding_sequence(input_data, n_in)
        input_data = word_index(input_data)
        target = predict_sequence(infenc, infdec, input_data, n_out, encoded_length)
        int_target = onehot_to_int(target)
        ans = invert(int_target)
        ans = ans.strip()
        #print(ans)

        datafile = "text/message2.txt"
        dataX = []
        dataY = []
        dataX, dataY = load_data(datafile)
        i = 0
        for i in range(len(dataX)):
            if dataX[i] == ans:
                print(dataY[i])
        #for sentence in dataY:
        #    print(sentence)
    return 0




def onehot_to_int(inputvector):
    return np.argmax(inputvector, axis=1)


def invert(inputlist):
    sentence = []
    for w in inputlist:
        sentence.append(idx2word(w))
    return "".join(sentence).replace("ว่างเปล่า", " ")


def idx2word(w):
    return wv_model.wv.index_to_key[w]


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array(word_index("_"))
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = np.array([[np.argmax(yhat[0, 0, :])]])
    return array(output)


if __name__ == '__main__':
    #train()
    test_model()
