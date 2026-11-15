import collections, datetime, hashlib, json, random, re, unicodedata
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = Path(r'C:\Users\YALAMA\projects\ml-frameworks-course-project\text-model\data\manual_candidates')
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
SHAS = {'hendzh/PromptShield':'a5234cb1f5cdb256600cab64b8c961195b5e8404', 'microsoft/llmail-inject-challenge':'1063bdf01ec8762b812d5e06ee768a06faa5a6f7','leolee99/NotInject':'847ae76cf8fea5ed325429e569ae8cfef022d2e0','deepset/prompt-injections':'4f61ecb038e9c3fb77e21034b22511b523772cdd'}
def get(url, **kw):
    r=session.get(url,timeout=90,**kw); r.raise_for_status(); return r
def norm(x): return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC',x).casefold()).strip()
def rows(ds, split, limit=None):
    offset=0
    while limit is None or offset < limit:
        j=get('https://datasets-server.huggingface.co/rows',params={'dataset':ds,'config':'default','split':split,'offset':offset,'length':100}).json()
        rr=j['rows']
        if not rr: break
        for r in rr: yield r
        offset+=len(rr)
        if offset>=j['num_rows_total']: break
print('Downloading original PromptShield splits',flush=True)
baseline=[]
for split in ['train','validation','test']:
    url=f'https://huggingface.co/datasets/hendzh/PromptShield/resolve/{SHAS["hendzh/PromptShield"]}/{split}.json'
    arr=get(url).json(); baseline.extend((norm(x['prompt']),split) for x in arr)
base_exact={x for x,_ in baseline}; print('baseline',len(baseline),flush=True)
base_by_len=[(x,s) for x,s in baseline if len(x)>=40]
items=[]
for split in ['NotInject_one','NotInject_two','NotInject_three']:
    for e in rows('leolee99/NotInject',split):
        x=e['row']; t=x['prompt'];
        if re.search(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]',t): continue
        items.append(dict(source='leolee99/NotInject',split=split,row=e['row_idx'],raw=t,text=t,label=0,context='',category=x.get('category',''),license='MIT'))
for split in ['train']:
    for e in rows('deepset/prompt-injections',split):
        x=e['row']; t=x['text']; label=x['label']
        if not isinstance(t,str) or label not in (0,1) or len(t)>1400: continue
        items.append(dict(source='deepset/prompt-injections',split=split,row=e['row_idx'],raw=t,text=t,label=label,context='',category='',license='Apache-2.0 (top-level card; metadata conflict review)',))
for e in rows('microsoft/llmail-inject-challenge','Phase1',limit=1000):
    x=e['row']; body=x.get('body') or ''; subj=x.get('subject') or ''
    if not x.get('scenario','').startswith('level1') or not (45<=len(body)<=900): continue
    if not re.search(r'\b(send|email|mail)\b',body,re.I): continue
    context='Summarize all following emails.'
    text=context+'\n\nEmail to summarize:\nSubject: '+subj+'\nBody: '+body
    items.append(dict(source='microsoft/llmail-inject-challenge',split='Phase1',row=e['row_idx'],raw=body,text=text,label=1,context=context,category=x.get('scenario',''),license='MIT',source_id=x.get('RowKey','')))
print('candidate pool',len(items),flush=True)
rng=random.Random(52); rng.shuffle(items); chosen=[]; seen=set(); counts=collections.Counter(); excluded=collections.Counter()
targets={('leolee99/NotInject',0):65,('deepset/prompt-injections',0):28,('deepset/prompt-injections',1):25,('microsoft/llmail-inject-challenge',1):32}
for it in items:
    key=(it['source'],it['label'])
    if counts[key]>=targets[key]: continue
    n=norm(it['raw']); full=norm(it['text'])
    if not n or n in seen or full in seen: excluded['selection_duplicates']+=1; continue
    if n in base_exact or full in base_exact: excluded['baseline_exact']+=1; continue
    if len(n)>=40 and any(n in b or b in n for b,_ in base_by_len): excluded['baseline_substring']+=1; continue
    # Avoid minor near-duplicates within the candidate selection.
    tokens=set(re.findall(r'\w+',n)); dup=False
    for prior in chosen:
        pt=set(re.findall(r'\w+',norm(prior['raw'])))
        if min(len(tokens),len(pt))>=5 and len(tokens & pt)/max(1,len(tokens | pt))>=0.85: dup=True; break
    if dup: excluded['candidate_near_duplicate']+=1; continue
    seen.update((n,full)); chosen.append(it); counts[key]+=1
print('selected',len(chosen),counts,'excluded',excluded,flush=True)
if counts != collections.Counter(targets): raise RuntimeError('insufficient candidate counts')
rng.shuffle(chosen)
OUT.mkdir(parents=True,exist_ok=False)
annot=[]; manifest=[]
for i,it in enumerate(chosen,1):
    ident=f'TXT-{i:03d}'
    annot.append({'id':ident,'text':it['text'],'label':[]})
    manifest.append({'id':ident,'source_dataset':it['source'],'source_revision':SHAS[it['source']],'source_split':it['split'],'source_row':it['row'],'source_record_id':it.get('source_id',''),'source_url':f'https://huggingface.co/datasets/{it["source"]}/tree/{SHAS[it["source"]]}','license':it['license'],'collected_on':'2026-11-15','source_label_candidate':it['label'],'category_or_scenario':it['category'],'context_note':('Scenario 1 user query; excerpt uses attack email only; full other email absent' if it['source'].startswith('microsoft/') else 'Original prompt; task or instruction hierarchy may be absent'),'annotation_status':'unreviewed'})
for name,seq in [('annotation_input.jsonl',annot),('provenance_private.jsonl',manifest)]:
    with (OUT/name).open('w',encoding='utf8',newline='\n') as f:
        for obj in seq: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
stats={'total':len(chosen),'targets':{'class_0':93,'class_1':57},'source_counts':{str(k):v for k,v in counts.items()},'excluded':dict(excluded),'baseline_rows':len(baseline),'baseline_revision':SHAS['hendzh/PromptShield'],'seed':52}
(OUT/'selection_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
