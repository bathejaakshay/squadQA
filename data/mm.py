import json
with open('dev-v2.0.json','r') as file, open('gold_test-v2.0.json','w') as output:
    final = {'qas':[]}
    it = json.load(file)
    for i in it['data']:
        for j in i['paragraphs']:
            for k in j['qas']:
                qa = {'question' : "", 'answers':[]}
                qa['answers'] = k['answers']
                qa['question'] = k['question']
                final['qas'].append(qa)
    print(final['qas'][:10])
    json.dump(final, output)