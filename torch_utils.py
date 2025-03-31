import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
#import numpy as np
import torch.optim as optim
import torch.nn.utils.rnn as rnn
#from tensorboardX import SummaryWriter
from datetime import datetime
from torchvision import models
import torch.nn.functional as fn
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from six.moves import cPickle as pickle
from torchvision.datasets.folder import accimage_loader
import argparse
import transformers
from transformers import BertTokenizer, BertModel
#from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F

#print(torch.__version__)
"""
def pil_loader(path): # PIL Loader is simply RGB
    print('Hi from inside pil loader')
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img

def default_loader(path): # accimage loader is faster but does not have all operations as PIL, uses intel IPP library
    print('Hi from inside default loader')
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        print('accimage')
        return accimage_loader(path)
    else:
        return pil_loader(path)
"""

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, num_embeddings, num_classes, embed_dim=768, k=30):#num_embeddings = len(q2i), num_classes = 1000, why embed_dim is 512?
        super().__init__()
        # nn. Embedding: The input is a list of numbers, and the output is a list of corresponding symbol embedding vectors
        # num_embeddings: The size of the dictionary, for example, if there are 5000 words in total, then enter 5000
        # embed_dim: The dimension of the input vector, that is, how many dimensions are used to represent a symbol
        #with open('./untitled/embedding_weights2.pkl', 'rb') as f:
        #    embedding_weights = pickle.load(f)
        #self.embed = nn.Embedding.from_pretrained(embedding_weights)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
        #self.bert_model = XLMRobertaModel.from_pretrained("xlm-roberta-base", output_hidden_states = True,)
        #self.embed = nn.Embedding(num_embeddings, embed_dim) # len(q2i) number of rows for words, 512 vector length or the number of columns
                                    #    512,      512 kernel size
        #print('Initial embedding is ', self.embed.weight)
        self.unigram_conv = nn.Conv1d(embed_dim, embed_dim, 1, stride=1, padding=0)
        #print('Initial unigram_conv  is ', self.unigram_conv.weight)
        self.bigram_conv  = nn.Conv1d(embed_dim, embed_dim, 2, stride=1, padding=1, dilation=2) #bigram means each KS=2
        # so each word will be convoluted twice, for the first and last word to be convoulted twice means we need an extra 1 word padding on each side
        # padding is usually kernel -1 # why dilation is 2? try with dilation = 1
        #print('Initial bigram_conv is ', self.bigram_conv.weight)
        self.trigram_conv = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding=2, dilation=2)# same as above trigram
        # so KS=3 and stride 1 means each word will be convoluted three times, for each word to be convoluted three times then we need 2 extra padding on each side
        #print('Initial trigram_conv is ', self.trigram_conv.weight)
        self.max_pool = nn.MaxPool2d((3, 1)) # kernel =3,1 which is equivalent to 3 rows and 1 column to get the max value of each word t in question q
        # Maximum pooling layer: The maximum pool is used in this article to obtain phrase-level characteristics
        # After maximum pooling, lstm is used to encode the sequence problem phrase level, and the corresponding problem-level feature qst is the hidden vector of time tLSTM.
        #print('Initial MaxPool  is ', self.max_pool)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=3, dropout=0.4) # "Vanilla LSTM" 3 layers
        #print('Initial lstm is ', self.lstm)
        # mean single input layer, single hidden layer and single output layer
        self.tanh = nn.Tanh()
        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        #print('Initial W_b is ', self.W_b)
        self.W_v = nn.Parameter(torch.randn(k, embed_dim)) # why k =30????????????????????????????
        #print('Initial W_v is ', self.W_v)
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        #print('Initial W_q is ', self.W_q)
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        #print('Initial w_hv is ', self.w_hv)
        self.w_hq = nn.Parameter(torch.randn(k, 1))
        #print('Initial w_hq is ', self.w_hq)

        #self.W_w = nn.Parameter(torch.randn(embed_dim, embed_dim))
        #self.W_p = nn.Parameter(torch.randn(embed_dim*2, embed_dim))
        #self.W_s = nn.Parameter(torch.randn(embed_dim*2, embed_dim))

        self.W_w = nn.Linear(embed_dim, embed_dim)
        #print('W_w is', self.W_w.weight)
        self.W_p = nn.Linear(embed_dim*2, embed_dim)
        self.W_s = nn.Linear(embed_dim*2, embed_dim) # why Ws for sentence or question is not embed_dim*3?

        self.fc = nn.Linear(embed_dim, num_classes)



    def forward(self, image, question):                    # Image: B x 512 x 196 # is the question here passed after
        # one-hot-encoding? question is a long tensor of 8 long numbers representing the question words
        #print('question type is ', type(question))
        #print('question is ', question)
        #print('question size is ', question.size())
        print('sorted tensors by length')
        #print(sorted(question, key=lambda x: x.size()[0], reverse=True))
        question, lens = rnn.pad_packed_sequence(question)
        #print('question after padding is ', question)
        #print('question after padding size is ', question.size())
        question = question.permute(1, 0)                  # Ques : B x L where B stands for batch and batch is 100
        #print('question after permute is  ', question)
        #print('question after permute size is ', question.size())
        #      question after permute size is  torch.Size([100, 17])
        #words = self.embed(question).permute(0, 2, 1)      # Words: B x L x 512


        segments_ids = []
        for i in range(len(question)):
          segments_ids.append([1]*len(question[0]))

        for x in range(len(question)):
          for y in range(len(question[0])):
            if (question[x][y] == 0):
              segments_ids[x][y] = 0

        segments_tensors = torch.tensor(segments_ids)
        #segments_tensors = segments_tensors.to(self.DEVICE)
        token_type_ids = []
        for i in range(len(question)):
          token_type_ids.append([0]*len(question[0]))
        token_type_ids_tensors = torch.tensor(token_type_ids)
        #token_type_ids_tensors = token_type_ids_tensors.to(self.DEVICE)

        self.bert_model.eval()
        with torch.no_grad():
          #self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
          words_c = self.bert_model(question,segments_tensors, token_type_ids_tensors)
          words = words_c['last_hidden_state'].permute(0, 2, 1)

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2) # B x 512 x L
        #print('unigrams size is ', unigrams.size())
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)  # B x 512 x L
        #print('bigrams size is ', bigrams.size())
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2) # B x 512 x L
        #print('trigrams size is ', trigrams.size())
        words = words.permute(0, 2, 1)
        #print('words after convolution and unsqueeze and permute size is ', words.size())
        #      words after convolution and unsqueeze and permute size is  torch.Size([100, 13, 512])

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))#,2) #####????????????? second 2
        phrase = torch.unsqueeze(phrase, 0) #####?????????????????????????????????????
        #print('phrase before permute size is ', phrase.size())
        phrase = phrase.permute(0, 2, 1)                                    # B x L x 512
        #print('phrase after permute size is ', phrase.size())

        hidden = None
        phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)                          # B x L x 512

        #print('image tensor size is ', image.size())
        #print('question tensor size is ', words.size())
        #print('phrase tensor size is ', phrase.size())
        #print('sentence tensor size is ', sentence.size())
        image = F.pad(input=image, pad=(0, 0, 128, 128), mode='constant', value=0)
        v_word, q_word = self.parallel_co_attention(image, words)
        v_word = v_word.unsqueeze(0)
        q_word = q_word.unsqueeze(0)
        #print('v_word size is ', v_word.size())
        #print('q_word size is ', q_word.size())
        v_phrase, q_phrase = self.parallel_co_attention(image, phrase)
        v_phrase = v_phrase.unsqueeze(0)
        q_phrase = q_phrase.unsqueeze(0)
        #print('v_phrase size is ', v_phrase.size())
        #print('q_phrase size is ', q_phrase.size())
        v_sent, q_sent = self.parallel_co_attention(image, sentence)
        v_sent = v_sent.unsqueeze(0)
        q_sent = q_sent.unsqueeze(0)
        #print('v_sent size is ', v_sent.size())
        #print('q_sent size is ', v_sent.size())

        #h_w = self.tanh(torch.matmul((q_word + v_word), self.W_w))
        #h_p = self.tanh(torch.matmul(torch.cat(((q_phrase + v_phrase), h_w), dim=1), self.W_p))
        #h_s = self.tanh(torch.matmul(torch.cat(((q_sent + v_sent), h_p), dim=1), self.W_s))

        h_w = self.tanh(self.W_w(q_word + v_word))
        #print('h_w size is ', h_w.size())
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        #print('h_p size is ', h_p.size())
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))
        #print('h_s size is ', h_s.size())

        logits = self.fc(h_s)

        return logits


    """
    def forward(self, image, question):                    # Image: B x 512 x 196 # is the question here passed after
        # one-hot-encoding? question is a long tensor of 8 long numbers representing the question words
        #print('question is ', question)
        #print('question type before padding is ', type(question))
        #print('question size is ', question.size())
        #print('sorted tensors by length')
        #print(sorted(question, key=lambda x: x.size()[0], reverse=True))
        #question, lens = rnn.pad_packed_sequence(question)
        #question = F.pad(input=question, pad=(0,512-list(question.size())[1], 0,0), mode='constant', value=0)
        print('inside forward')
        #print('image tensor is ', image)
        print('question type after padding is ', type(question))
        print('question after padding is ', question)
        print('question after padding size is ', question.size())
        #question = question.permute(1, 0)                  # Ques : B x L where B stands for batch and batch is 100
        print('question after permute is ', question)
        print('question after permute size is ', question.size())
        #      question after permute size is  torch.Size([100, 17])
        words = self.embed(question).permute(0,2,1)      # Words: B x L x 512
        print('words/question after embedding type is ', type(words))
        #delete from here
        #words = self.embed(question)
        print('words after embed and permute size is ', words.size())
        #print('words is ', words)
        #words = words.permute(0,2,1)
        #print('words size after permute is ', words.size())
        # till here

        #torch.set_printoptions(profile="full")
        #print('words is ', words)
        #print('words after embed and permute size is ', words.size())

        #      words after embed and permute size is torch.Size([100, 512, 13])
        #unigrams size is torch.Size([100, 512, 1, 13])
        #bigrams size is torch.Size([100, 512, 1, 13])
        #trigrams size is torch.Size([100, 512, 1, 13])

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2) # B x 512 x L
        #unigrams = self.tanh(self.unigram_conv(words))
        print('unigrams size is ', unigrams.size())
        #print('unigrams class is ', type(unigrams))
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)  # B x 512 x L
        print('bigrams size is ', bigrams.size())
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2) # B x 512 x L
        print('trigrams size is ', trigrams.size())
        words = words.permute(0,2,1)
        print('words after convolution and unsqueeze and permute size is ', words.size())
        #      words after convolution and unsqueeze and permute size is  torch.Size([100, 13, 512])
        #print('words is ', words)


        phrase = torch.cat((unigrams, bigrams, trigrams),2)
        print('phrase after torch cat size is ', phrase.size())
        phrase= self.max_pool(phrase)
        print('phrase after maxpool size is ', phrase.size())
        print('phrase is ', phrase)
        phrase = torch.squeeze(phrase,2)
        print('phrase after squeeze and before permute size is ', phrase.size())
        print('phrase is ', phrase)

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.unsqueeze(0) # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        #phrase = self.max_pool(torch.cat((unigrams, bigrams, trigrams),2))
        #phrase = phrase.squeeze(2)
        #phrase = torch.squeeze(phrase, 2)
        print('phrase before permute size is ', phrase.size())
        #print('phrase after max pool and squeeze is ', phrase)
        phrase = phrase.permute(0, 2, 1)                                    # B x L x 512
        #phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        #phrase = phrase.permute(0, 2, 1)
        #phrase = phrase.permute(0,2,1)                                    # B x L x 512
        print('phrase size is ', phrase.size())

        hidden = None
        #phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        phrase_packed = torch.transpose(phrase, 0, 1)
        #sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        #sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence, hidden = self.lstm(phrase_packed, hidden)
        #sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)
        #sentence = torch.transpose(sentence_packed, 0, 1)                          # B x L x 512


        print('image tensor size is ', image.size())
        print('question tensor size is ', words.size())
        print('phrase tensor size is ', phrase.size())
        print('sentence tensor size is ', sentence.size())

        v_word, q_word = self.parallel_co_attention(image, words)
        v_word = v_word.unsqueeze(0)
        q_word = q_word.unsqueeze(0)
        print('v_word size is ', v_word.size())
        print('q_word size is ', q_word.size())
        v_phrase, q_phrase = self.parallel_co_attention(image, phrase)
        v_phrase = v_phrase.unsqueeze(0)
        q_phrase = q_phrase.unsqueeze(0)
        print('v_phrase size is ', v_phrase.size())
        print('q_phrase size is ', q_phrase.size())
        v_sent, q_sent = self.parallel_co_attention(image, sentence)
        v_sent = v_sent.unsqueeze(0)
        q_sent = q_sent.unsqueeze(0)
        print('v_sent size is ', v_sent.size())
        print('q_sent size is ', v_sent.size())

        #h_w = self.tanh(torch.matmul((q_word + v_word), self.W_w))
        #h_p = self.tanh(torch.matmul(torch.cat(((q_phrase + v_phrase), h_w), dim=1), self.W_p))
        #h_s = self.tanh(torch.matmul(torch.cat(((q_sent + v_sent), h_p), dim=1), self.W_s))

        h_w = self.tanh(self.W_w(q_word + v_word))
        print('h_w size is ', h_w.size())
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        print('h_p size is ', h_p.size())
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))
        print('h_s size is ', h_s.size())

        logits = self.fc(h_s)
        print('logits type is ', type(logits))
        print('logits size is ', logits.size())

        return logits
    """

    def parallel_co_attention(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x 196
        #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        #print('size of a_v is ', a_v.size())
        #print('size of a_q is ', a_q.size())

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))#,1) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))#,1)                  # B x 512

        return v, q

q2i_len = 15196
num_classes = 1000
print('Just before CoattentionNet')
model = CoattentionNet(q2i_len, num_classes).float()
print('Just after CoattentionNet')
#PATH = "./untitled/checkpoint_25_Ahmed.pth.tar"
PATH = "app/checkpoint_17_Ahmed_768_new.pth.tar"
#PATH = "app/checkpoint_26_Ahmed_768_new_XLMBERTa.pth.tar"
#PATH = "https://drive.google.com/file/d/1-g04V9ntMu2g5r3N_U0RaJSmlmbdeRcw/view?usp=share_link"
print('loding the model')
model.load_state_dict(torch.load(PATH, map_location='cpu'))
print('loding the model successful')


#print('done reloding')
 # Use the GPU if it's available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    #print('Device is cuda')
    self._model = self._model.cuda()
#model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()
#print('model W_b is', model.W_b)
#print('model W_w is ', model.W_w.weight)
#print('model parameters are ',list(model.parameters()))
#image -> tensor

def transform_image(img_name):
    #print('inside transform_image')
    transform = transforms.Compose([transforms.Resize((448, 448)),
                                    transforms.ToTensor()])

    #img_name = img
    #img = default_loader(img_name)
    imgT = transform(img_name).float()
    imgT = imgT.unsqueeze(0)
    #print('image tensor size after transform is ', imgT.size())

    #print('Creating Image Encoder')
    img_enc = models.resnet18(pretrained=True)
    modules = list(img_enc.children())[:-2]
    #img_enc = models.resnet18(pretrained=True)
    #modules = list(img_enc.children())[:-2] # resnet feature extractor, all layer except last 2 or from layer 0 to layer 7, can use [0:8] which means 0 --> 7
    img_enc = nn.Sequential(*modules)
    for params in img_enc.parameters():
            params.requires_grad = False
    #if DEVICE == "cuda":
        #img_enc = img_enc.cuda()
    img_enc.eval()

    #print('1- image tensor size is ', imgT.size())
    #imgT = imgT.to(self.DEVICE)
    #print('before img_enc')
    imgT = img_enc(imgT)
    #print('3- image tensor size is ', imgT.size())
    imgT = imgT.view(imgT.size(0), imgT.size(1), -1)
    print('4- image tensor size is ', imgT.size())

    #imgT2 = imgT.repeat(2,1,1)#tf.tile(imgT, [2,1])
    #print('imgT2 is ', imgT2)
    #print('size of imgT2 is ', imgT2.size())

    #imgT = img_enc(imgT)
    #imgT = imgT.view(imgT.size(0), imgT.size(1), -1)
    #image = Image.open(io.BytesIO(image_bytes))
    #return transform(image).unsqueeze(0)
    #print('done transform_image')
    return imgT#.float() # unsqueeze(0)?
    #return imgT.unsqueeze(0)


"""
# question -> tensor
def transform_question(question):
    print('inside transform_question')
    with open('q2i.pkl', 'rb') as f:
        q2i = pickle.load(f)
    #qqa = question.split()
    print('length of q2i is ', len(q2i))
    #print('q2i is ', q2i)
    q2i_keys = q2i.keys()
    ques = question[:-1] #Get the question with the question mark removed
    quesI = [q2i["<sos>"]] + [q2i[x.lower()] for x in ques.split(" ") if x.lower() in q2i_keys] + [q2i["<eos>"]]
    #quesI = quesI + [q2i["<pad>"]]*(20 - len(quesI))
    # q2i={dict:11471}, q2i={"<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3, 'what':4, 'color':5. . . }
    # quesI = [1,4,454,54,843,6,11,2]
    #if not self.collate:
    #quesI = quesI + [q2i["<pad>"]]*(8 - len(quesI)) #????? why 8? What if the question len(quesI greater than 8)

    quesT = torch.from_numpy(np.array(quesI)).long()
    print('quesT before packing is ', quesT)
    print('quesT size before packing is ', quesT.size())
    #quesT = rnn.pack_sequence(quesT) # try without it
    print('question tensor size is ', quesT.size())
    print('done transform_question ', quesT)

    return quesT.unsqueeze(0)
"""

def transform_question_BERT(question):
    ques = question[:-1]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    tokens = tokenizer(ques)
    indexes = (tokens['input_ids'])
    quesT = torch.tensor([indexes]).squeeze(0)
    print('BERT quesT before packing is ', quesT)
    print('BERT quesT size is ', quesT.size())
    quesT_list = []
    quesT_list.append(quesT)
    quesT = rnn.pack_sequence(quesT_list)
    print('BERT done transform_question ', quesT)
    return quesT#.unsqueeze(0)

"""
def transform_question_two(question):
    print('inside transform_question_twoooooooooooooooooooooooooooo')
    with open('q2i.pkl', 'rb') as f:
        q2i = pickle.load(f)
    #qqa = question.split()
    print('length of q2i is ', len(q2i))
    #print('q2i is ', q2i)
    q2i_keys = q2i.keys()
    ques = question[:-1] #Get the question with the question mark removed
    quesI = [q2i["<sos>"]] + [q2i[x.lower()] for x in ques.split(" ") if x.lower() in q2i_keys] + [q2i["<eos>"]]
    #quesI = quesI + [q2i["<pad>"]]*(20 - len(quesI))
    # q2i={dict:11471}, q2i={"<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3, 'what':4, 'color':5. . . }
    # quesI = [1,4,454,54,843,6,11,2]
    #if not self.collate:
    #quesI = quesI + [q2i["<pad>"]]*(8 - len(quesI)) #????? why 8? What if the question len(quesI greater than 8)

    quesT = torch.from_numpy(np.array(quesI)).long()
    print('quesT before packing is ', quesT)
    print('type is ', quesT.type())
    print('quesT size before packing is ', quesT.size())
    quesT_list = []
    quesT_list.append(quesT)
    print('first append')
    #fake = torch.zeros(9)
    #quesT_list.append(fake)
    #quesT = quesT.tolist()

    print('quesT rows is ', len(quesT_list))
    print('quesT column is ', len(quesT_list[0]))
    print('questT type is ', type(quesT_list))
    print('changed to list trying to pack', type(quesT_list))
    print('quesT before pack is ', quesT_list)
    quesT = rnn.pack_sequence(quesT_list)
    #quesT = rnn.pack_sequence(quesT) # try without it
    #print('question tensor size is ', quesT.size())
    print('done transform_question ', quesT)

    return quesT
"""

# predict

def get_prediction(image_tensor,question_tensor):
    print('inside get_prediction')
    #images = image_tensor.reshape(-1, 28*28)
    torch.set_printoptions(profile="full")
    #print('image tensor is ', image_tensor)
    predicted_answer = model(image_tensor,question_tensor)
    print('predicted answer using torchmax is ', torch.argmax(predicted_answer).item())
    print('predicted answer size is ', predicted_answer.size())
    #print('real prediction is ', predicted_answer[0][0].item())
    #print('real prediction is ', predicted_answer[0][1].item())
    total=0
    biggest =0
    index=0
    for i in range(1000):
        total = total + predicted_answer[0][i].item()
        if predicted_answer[0][i].item() > biggest :
            biggest=predicted_answer[0][i].item()
            index=i
    print ('total is ', total)
    print ('biggest is ', biggest)
    print('index is ', index)

        # max returns (value ,index)
    prob, predicted = torch.max(predicted_answer, 1)
    print('top answer probability is ', prob.item())
    print('predicted class is ', predicted.item())
    prob_list, predicted_list = torch.sort(predicted_answer, 1, descending=True)
    print('top first predictions list is ', prob_list[0][0].item(), ' and class is ', predicted_list[0][0].item())
    print('top second predictions list is ', prob_list[0][1])
    print('top third predictions list is ', prob_list[0][2])
    print('top fourth predictions list is ', prob_list[0][3])
    print('top fifth predictions list is ', prob_list[0][4])
    print('top sixth predictions list is ', prob_list[0][5])
    print('top seventh predictions list is ', prob_list[0][6])
    print('top eighth predictions list is ', prob_list[0][7])
    print('top ninth predictions list is ', prob_list[0][8])
    print('top tenth predictions list is ', prob_list[0][9])
    #print('predicted type is ', predicted.type())
    #print('prediction list type is ', predicted_list.type())

    #print('prediction list is ', predicted_list)
    #top_three = tf.gather(predicted_list, [0, 1, 2])
    #print('top 3 predictions list is ', predicted_list[0][0])
    #print('top 3 predictions list is ', predicted_list[0][1])
    #print('top 3 predictions list is ', predicted_list[0][2])
    #print('done with prediction')
    return predicted_list
