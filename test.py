import flask
import random
from flask import send_from_directory, request
import pandas as pd
# import Flask from flask module
from nltk.corpus import stopwords
import pickle
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import nltk

# nltk.download('all')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# !pip install transformers
# !pip install -U transformers
# !pip install sentencepiece

app = flask.Flask(__name__)
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")


@app.route("/classification", methods=["POST", "GET"])
def classify():
    inp = request.get_json(force=True)
    action = inp['queryResult']['action']
    parameter = inp['queryResult']["parameters"]
    print(f"parameter : {parameter}")
    print(f"action : {action}")

    print(inp)
    # if action =='summarize':

    # Create tokens - number representation of our text
    long_text = parameter["long_text"]

    # Tokenize and preprocess the input text
    tokens = tokenizer(
        long_text,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    # Generate the summary
    generated_summary = model.generate(**tokens)

    # Convert token IDs to text summary
    decoded_summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)

    return {"fulfillmentText": f"{decoded_summary}"}


if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()
