

class Corpus:
    def __init__(self, dataframe, label):
        self.corpus = dataframe[label]
        self.tokenized_corpus = self.tokenize()
        self.clean_vocabulary = self.extract_voc()

    def tokenize(self, sep=' '):
        tokenized_corpus = []
        for sentence in self.corpus:
            tokenized_sentence = []
            for token in sentence.split(sep):
                clean_token = self.clean_token(token)
                tokenized_sentence += clean_token
            tokenized_corpus.append(tokenized_sentence)
        return tokenized_corpus

    def extract_voc(self):
        vocabulary = []
        for sentence in self.tokenized_corpus:
            for token in sentence:
                if token not in vocabulary:
                    vocabulary.append(token)
        return vocabulary

    def clean_vocabulary(self):
        clean_voc = []
        for element in self.vocabulary:
            clean_element = self.clean_token(element)
            for chunk in clean_element:
                if chunk not in clean_voc:
                    clean_voc.append(chunk)
        return clean_voc

    def clean_token(self, token):
        clean_token = token.lower()
        if len(clean_token) < 2:
            return clean_token
        out_char = ['.', ',', '?', '!', ";", "#", "\"", "*", "%", "-", "_", "+", "/" "=", "@", "(", ")", "’", "”", "'"]
        chunk_1 = [clean_token]
        chunk_2 = []
        for sep in out_char:
            for chunk in chunk_1:
                chunk_2 += chunk.split(sep)
            chunk_1 = chunk_2
            chunk_2 = []
        output = []
        for element in chunk_1:
            if len(element) > 0:
                '''
                if element[-2:] == "\''s":
                    output.append(element[:-2])
                    output.append("is")
                elif element[-3:] == "\'re":
                    output.append(element[:-3])
                    output.append("are")
                elif element[-2:] == "\'m":
                    output.append(element[:-2])
                    output.append("am")
                elif element[0] == "\"":
                    output.append(element[1:])
                elif element[-1] == "\"":
                    output.append(element[:-1])
                else:
                '''
                output.append(element)
        return output


