import torch
import pickle
from torch.utils.data import DataLoader
from model import GRULanguageModel

def generate_sentence_from_bos(model, vocab, bos=1):
    '''
        Input Parameters
        - bos: begin-of-sentence token index. usually 1
        Output returns
        - generated_sentence: a sentence generated by the model
        Example
            >>> import pickle
            >>> import torch
            >>> from model import GRULanguageModel
            >>> from generate import generate_sentence_from_bos
            >>> vocab = pickle.load(open('vocab.pickle', 'rb'))
            >>> hidden_size = 30
            >>> output_size = len(vocab)
            >>> model = GRULanguageModel(hidden_size=hidden_size, output_size=output_size)
            >>> model.load_state_dict(torch.load('gru_model.bin'))
            <All keys matched successfully>
            >>> model.eval()
            GRULanguageModel(
              (embedding): Embedding(21, 30)
              (gru): GRU(30, 30, batch_first=True)
              (softmax): LogSoftmax(dim=-1)
              (out): Linear(in_features=30, out_features=21, bias=True)
            )
            >>> generated_text = generate_sentence_from_bos(model, vocab, bos=1)
            >>> print('generated sentence: {}'.format(generated_text))
            generated sentence: <s> for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    '''
    indice = [bos]
    hidden = torch.zeros((1, 1, model.hidden_size))
    lm_inputs = torch.tensor(indice).unsqueeze(-1)
    i2v = {v:k for k, v in vocab.items()}

    cnt = 0
    eos = vocab['</s>']
    generated_sequence = [lm_inputs[0].data.item()]
    while True:
        if cnt == 30:
            break
        output, hidden = model(lm_inputs, hidden)
        output = output.squeeze(1)
        topv, topi = output.topk(1)
        lm_inputs = topi

        if topi.data.item() == eos:
            tokens = list(map(lambda w: i2v[w], generated_sequence))
            generated_sentence = ' ' .join(tokens)
            return generated_sentence

        generated_sequence.append(topi.data.item())
        cnt += 1

    print('max iteration reached. therefore finishing forcefully')
    tokens = list(map(lambda w: i2v[w], generated_sequence))

    generated_sentence = ' ' .join(tokens)
    return generated_sentence

def main():
    # define dataset and dataloader
    vocab = pickle.load(open('vocab.pickle', 'rb'))

    # define and load model
    hidden_size = 30
    output_size = len(vocab)
    model = GRULanguageModel(hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load('gru_model.bin'))
    model.eval()

    # Generate a text
    generated_text = generate_sentence_from_bos(model, vocab, bos=1)
    print('generated sentence: {}'.format(generated_text))

if __name__ == '__main__':
    main()