import argparse
import collections
import difflib
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import requests
from langdetect import DetectorFactory, LangDetectException, detect
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DetectorFactory.seed = 52
SEED = 52
TOTAL = 200
SHAS = {
    'hendzh/PromptShield': 'a5234cb1f5cdb256600cab64b8c961195b5e8404',
    'leolee99/NotInject': '847ae76cf8fea5ed325429e569ae8cfef022d2e0',
    'deepset/prompt-injections': '4f61ecb038e9c3fb77e21034b22511b523772cdd',
}
TARGETS = {('leolee99/NotInject', 0): 87, ('deepset/prompt-injections', 0): 37,
           ('deepset/prompt-injections', 1): 76}
EXCLUDED_DEEPSET_ROWS = {158, 377, 437, 469, 473, 488, 492, 499, 505, 513, 515, 516, 518, 519, 521, 528, 540, 545}
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

def get(url, **kw):
    response = session.get(url, timeout=90, **kw)
    response.raise_for_status()
    return response

def check_source_revisions():
    for dataset in ('leolee99/NotInject', 'deepset/prompt-injections'):
        actual = get('https://huggingface.co/api/datasets/' + dataset).json()['sha']
        if actual != SHAS[dataset]:
            raise RuntimeError(f'Source revision changed: {dataset}: {actual}')

def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()

def words(text):
    return set(re.findall(r'\w+', norm(text)))

def is_english(text):
    if any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text):
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def rows(dataset, split):
    offset = 0
    while True:
        page = get('https://datasets-server.huggingface.co/rows', params={
            'dataset': dataset, 'config': 'default', 'split': split,
            'offset': offset, 'length': 100,
        }).json()
        for row in page['rows']:
            yield row
        offset += len(page['rows'])
        if not page['rows'] or offset >= page['num_rows_total']:
            break

def similar(a, b):
    a, b = norm(a), norm(b)
    wa, wb = words(a), words(b)
    return (max(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(),
                difflib.SequenceMatcher(None, b, a, autojunk=False).ratio()) >= 0.75 or
            (min(len(wa), len(wb)) >= 5 and len(wa & wb) / len(wa | wb) >= 0.75))

def build_baseline_checker(baseline):
    base_exact = set(baseline)
    base_long = [x for x in base_exact if len(x) >= 40]
    base_words = [words(x) for x in baseline]
    word_index = collections.defaultdict(list)
    for i, tokens in enumerate(base_words):
        for token in tokens:
            word_index[token].append(i)
    def baseline_overlap(text):
        text = norm(text)
        if text in base_exact:
            return 'exact'
        if len(text) >= 40 and any(text in x or x in text for x in base_long):
            return 'substring'
        tokens = words(text)
        if len(tokens) >= 5:
            overlaps = collections.Counter(i for word in tokens for i in word_index[word])
            for i, common in overlaps.items():
                if common / (len(tokens) + len(base_words[i]) - common) >= 0.75:
                    return 'words'
        return None
    return baseline_overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Path to a new output directory')
    out = Path(parser.parse_args().output)
    if out.exists():
        raise FileExistsError(f'Output directory already exists: {out}')
    registry = json.loads((Path(__file__).parent / 'stable_id_map.json').read_text(encoding='utf8'))
    if len(registry) != TOTAL or [x['id'] for x in registry] != [f'TXT-{i:03d}' for i in range(1, TOTAL + 1)]:
        raise ValueError('Expected the 200 registered records in ID order')
    check_source_revisions()
    baseline = []
    print('Downloading PromptShield splits', flush=True)
    for split in ('train', 'validation', 'test'):
        url = f'https://huggingface.co/datasets/hendzh/PromptShield/resolve/{SHAS["hendzh/PromptShield"]}/{split}.json'
        baseline.extend(norm(x['prompt']) for x in get(url).json())
    baseline_overlap = build_baseline_checker(baseline)
    items = []
    for split in ('NotInject_one', 'NotInject_two', 'NotInject_three'):
        for e in rows('leolee99/NotInject', split):
            x = e['row']
            items.append(dict(source='leolee99/NotInject', split=split, row=e['row_idx'], text=x['prompt'],
                              label=0, category=x.get('category', ''), license='MIT'))
    for e in rows('deepset/prompt-injections', 'train'):
        x = e['row']
        if isinstance(x['text'], str) and x['label'] in (0, 1) and len(x['text']) <= 1400:
            items.append(dict(source='deepset/prompt-injections', split='train', row=e['row_idx'],
                              text=x['text'], label=x['label'], category='',
                              license='Apache-2.0 (top-level card; metadata conflict review)'))
    # Load only explicitly retained rows from the source test split. Keep them
    # outside the shuffled candidate pool so the remaining IDs stay unchanged.
    needed = {x['source_row'] for x in registry
              if x['source_dataset'] == 'deepset/prompt-injections' and x['source_split'] == 'test'}
    extra_items = []
    if needed:
        for e in rows('deepset/prompt-injections', 'test'):
            if e['row_idx'] in needed:
                x = e['row']
                extra_items.append(dict(source='deepset/prompt-injections', split='test', row=e['row_idx'],
                                        text=x['text'], label=x['label'], category='',
                                        license='Apache-2.0 (top-level card; metadata conflict review)'))
                if len(extra_items) == len(needed):
                    break
    check_source_revisions()
    key = lambda x: (x['source'], x['split'], x['row'])
    by_key = {key(item): item for item in items + extra_items}
    chosen = []
    existing_similarity_pairs = []
    counts = collections.Counter()
    for entry in registry:
        source_key = (entry['source_dataset'], entry['source_split'], entry['source_row'])
        item = by_key.get(source_key)
        if not item or hashlib.sha256(item['text'].encode('utf8')).hexdigest() != entry['text_sha256']:
            raise ValueError(f"Previous text changed or missing: {entry['id']}")
        if baseline_overlap(item['text']):
            raise ValueError(f"Previous text overlaps PromptShield: {entry['id']}")
        for prior_index, prior in enumerate(chosen, 1):
            if similar(item['text'], prior['text']):
                existing_similarity_pairs.append([f'TXT-{prior_index:03d}', entry['id']])
        chosen.append(item)
        counts[(item['source'], item['label'])] += 1
    used = {key(item) for item in chosen}
    random.Random(SEED).shuffle(items)
    excluded = collections.Counter()
    for item in items:
        group = (item['source'], item['label'])
        if counts[group] >= TARGETS[group] or key(item) in used:
            continue
        if item['source'] == 'deepset/prompt-injections' and item['row'] in EXCLUDED_DEEPSET_ROWS:
            excluded['manual_language'] += 1
            continue
        if not is_english(item['text']):
            excluded['language'] += 1
            continue
        overlap = baseline_overlap(item['text'])
        if overlap:
            excluded['baseline_' + overlap] += 1
            continue
        if any(similar(item['text'], prior['text']) for prior in chosen):
            excluded['candidate_near_duplicate'] += 1
            continue
        chosen.append(item)
        used.add(key(item))
        counts[group] += 1
    if len(chosen) != TOTAL or counts != collections.Counter(TARGETS):
        raise RuntimeError(f'Insufficient candidates: {len(chosen)}, {counts}')
    annot, manifest = [], []
    for i, item in enumerate(chosen, 1):
        ident = f'TXT-{i:03d}'
        annot.append({'id': ident, 'text': item['text'], 'label': []})
        manifest.append({'id': ident, 'source_dataset': item['source'], 'source_revision': SHAS[item['source']],
                         'source_split': item['split'], 'source_row': item['row'], 'source_record_id': '',
                         'source_url': f'https://huggingface.co/datasets/{item["source"]}/tree/{SHAS[item["source"]]}',
                         'license': item['license'], 'collected_on': '2026-11-15',
                         'source_label_candidate': item['label'], 'category_or_scenario': item['category'],
                         'context_note': 'Original prompt; task or instruction hierarchy may be absent',
                         'annotation_status': 'unreviewed'})
    out.mkdir(parents=True, exist_ok=False)
    for name, data in (('annotation_input.jsonl', annot), ('provenance_private.jsonl', manifest)):
        with (out / name).open('w', encoding='utf8', newline='\n') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    stats = {'total': len(chosen), 'targets': {'class_0': 124, 'class_1': 76},
             'source_counts': {str(k): v for k, v in counts.items()}, 'excluded': dict(excluded),
             'baseline_rows': len(baseline), 'baseline_revision': SHAS['hendzh/PromptShield'],
             'baseline_checks': ['exact', 'substring_min_40_chars', 'word_jaccard_0.75_min_5_words'],
             'existing_similarity_pairs': existing_similarity_pairs,
             'seed': SEED}
    (out / 'selection_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('selected', len(chosen), dict(counts), 'excluded', dict(excluded), flush=True)

if __name__ == '__main__':
    main()
