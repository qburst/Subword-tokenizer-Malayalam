from first_network.custom_model_inference import predict_split
from second_network.custom_model_inference import predict_subwords

def convert(word, split):
    index = split.index('1')
    text = list(word)
    text[index] = chr(ord(text[index]) + 150)
    text = ''.join(text)

    return text

def tokenize(word):
    split = predict_split(word)
        
    if '1' not in split:
        return word
    word = convert(word, split)

    subwords = predict_subwords(word)
    
    return subwords

if __name__=='__main__':
    if(len(sys.argv) > 1):
        print(tokenize(sys.argv[1]))
