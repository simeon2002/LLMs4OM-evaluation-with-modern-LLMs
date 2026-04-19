import json, glob, os
import pandas as pd

rows = []

files = glob.glob('experiments/outputs/bio-ml/*/*RAG*.json')
print(files)
for f in files:

    with open(f) as fp:
        d = json.load(fp)

    row = {
        'model': d.get('model'),
        'task': d.get('dataset-info', {}).get('ontology-name'),
        'encoder': d.get('encoder-id'),
    }

    evaluation_results = d.get('evaluation-results')
    if (evaluation_results is not None):
        r = evaluation_results.get('full')
        row['precision'] = r['precision']
        row['recall'] = r['recall']
        row['f-score'] = r['f-score']
    else:
        row['precision'] = 'N/A'
        row['recall'] = 'N/A'
        row['f-score'] = 'N/A'

    rows.append(row)
df = pd.DataFrame(rows)
print(df)

# reshaping the flat table to make it look better
df2 = df.set_index(['model', 'task', 'encoder'])
print(df2)

df3 = df2.unstack(['task', 'encoder'])
print(df3)

df3 = df3.reorder_levels(['task', 'encoder', None], axis=1).sort_index(axis=1)
print(df3)
