from transformers import BertTokenizer
from tokenizers import Tokenizer, Encoding
from tokenizers.models import WordPiece


tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased", proxies={"https": ""})
text = "ahoj tu som, a ty?"
tokenizer.tokenize(text)
tokenizer.encode(text)
