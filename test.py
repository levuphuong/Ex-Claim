# from phonlp import load

# model = load(save_dir="models/phonlp")

# text = "Tôi đang làm_việc tại VinAI ."
# annot = model.annotate(text=text)

# for sent in annot:
#     print("=== Sentence ===")
    
#     # 3 cases:
#     # 1. sent = [forms, poss, ners]
#     if len(sent) == 3 and isinstance(sent[0], list):
#         forms, poss, ners = sent
#         for f,p,n in zip(forms, poss, ners):
#             print(f"{f:15} POS={p:6} NER={n}")
    
#     # 2. sent = list of tuples
#     elif isinstance(sent, list) and isinstance(sent[0], tuple):
#         for tup in sent:
#             if len(tup) == 3:
#                 f,p,n = tup
#                 print(f"{f:15} POS={p:6} NER={n}")
#             elif len(tup) == 2:
#                 f,p = tup
#                 print(f"{f:15} POS={p:6} NER=O")
#             else:
#                 print(tup)
    
#     else:
#         print("Unknown format:", sent)
from phonlp import load
import phonlp

phonlp.download(save_dir="models/phonlp")
model = phonlp.load(save_dir="models/phonlp")
print(model.annotate(text="Tôi đang làm việc tại VinAI."))