from flask import Flask, render_template, request, send_file, redirect, url_for, session, jsonify
import re
import string
import uvicorn
from torch import clamp
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class TokenSimilarity:

    def load_pretrained(self, from_pretrained:str="indobenchmark/indobert-base-p1"):
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        self.model = AutoModel.from_pretrained(from_pretrained)
        
    def __cleaning(self, text:str):
        # clear punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # clear multiple spaces
        text = re.sub(r'/s+', ' ', text).strip()

        return text
        
    def __process(self, first_token:str, second_token:str):
        inputs = self.tokenizer([first_token, second_token],
                                max_length=self.max_length,
                                truncation=self.truncation,
                                padding=self.padding,
                                return_tensors='pt')

        attention = inputs.attention_mask

        outputs = self.model(**inputs)

        # get the weights from the last layer as embeddings
        embeddings = outputs[0] # when used in older transformers version
        # embeddings = outputs.last_hidden_state # when used in newer one

        # add more dimension then expand tensor
        # to match embeddings shape by duplicating its values by rows
        mask = attention.unsqueeze(-1).expand(embeddings.shape).float()

        masked_embeddings = embeddings * mask
        
        # MEAN POOLING FOR 2ND DIMENSION
        # first, get sums by 2nd dimension
        # second, get counts of 2nd dimension
        # third, calculate the mean, i.e. sums/counts
        summed = masked_embeddings.sum(1)
        counts = clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed/counts

        # return mean pooling as numpy array
        return mean_pooled.detach().numpy()
        
    def predict(self, first_token:str, second_token:str,
                return_as_embeddings:bool=False, max_length:int=16,
                truncation:bool=True, padding:str="max_length"):
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        first_token = self.__cleaning(first_token)
        second_token = self.__cleaning(second_token)

        mean_pooled_arr = self.__process(first_token, second_token)
        if return_as_embeddings:
            return mean_pooled_arr

        # calculate similarity
        similarity = cosine_similarity([mean_pooled_arr[0]], [mean_pooled_arr[1]])

        return similarity

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/upload',  methods=['POST'])
def process():
    model = TokenSimilarity()
    model.load_pretrained('indobenchmark/indobert-base-p2')

    answer = request.form.get('answer')

    option = {}
    option[0] = request.form.get('option1')
    option[1] = request.form.get('option2')
    option[2] = request.form.get('option3')
    option[3] = request.form.get('option4')

    result = {}
    result[0] = model.predict(request.form.get('answer'), request.form.get('option1'))
    result[1] = model.predict(request.form.get('answer'), request.form.get('option2'))
    result[2] = model.predict(request.form.get('answer'), request.form.get('option3'))
    result[3] = model.predict(request.form.get('answer'), request.form.get('option4'))

    return render_template('result.html', answer=answer, option=option, result=result)

if __name__ == '__main__':
    app.run(debug=True)
    uvicorn.run("main:app", host = "127.0.0.1", port = 5000, log_level = "info", reload = True)