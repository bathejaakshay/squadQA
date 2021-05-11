import json
with open('train-v2.0.json','r') as file:
    # final = {'qas':[]}
    it = json.load(file)
    print(len(it['data'][0]['paragraphs'][0]['qas']))