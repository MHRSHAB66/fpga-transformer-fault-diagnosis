"""Independent saved-evidence audit. Offline only; production artifacts are read only."""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
def load(p): return json.loads((ROOT / p).read_text(encoding='utf8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def save():
    (OUT / 'results.json').write_text(json.dumps(R, indent=2, default=lambda v: v.item()), encoding='utf8')
def rows(p):
    ans = []
    for line in Path(p).read_text().splitlines():
        if line.startswith('R '): ans.append(dict(x.split('=', 1) for x in line.split()[1:]))
    return ans
R = {'head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), 'board_accessed': False}
f = load('artifacts/research/F00_20261113/frozen_baseline_manifest.json')
R['frozen'] = {k: {'count':len(f[k]), 'changed':[x['path'] for x in f[k] if sha(ROOT/x['path']) != x['sha256']]}
               for k in ['model_and_export','hls_sources','golden','reference_outputs']}
R['bundles'] = {}
for d in sorted((ROOT/'artifacts/deploy').glob('research_*')):
    if not (d/'MANIFEST.sha256').exists(): continue
    ent = [l.split(' *',1) for l in (d/'MANIFEST.sha256').read_text().splitlines()]
    c = json.loads((d/'config/variant_contract.json').read_text())
    R['bundles'][d.name] = {'count':len(ent),'bad_hashes':[p for h,p in ent if sha(d/p.removeprefix('./')) != h],
                          'numeric':c['numeric']['attention_probabilities'], 'goldens':c['golden']['count']}
probe=subprocess.run([sys.executable,'tools/package_seq256_pynq_deploy.py','--check',
    '--pkg','artifacts/research/A06_20261113/a06_build','--bundle','artifacts/deploy/research_a06_z7020_20261113',
    '--golden','artifacts/research/A06_20261113/golden_raw1024_p8u',
    '--contract','artifacts/research/contracts/research_b04_z7020_20261114.json'],cwd=ROOT,capture_output=True,text=True)
R['checker_wrong_contract_probe']={'exit_code':probe.returncode,'stdout':probe.stdout.strip(),
                                   'incorrectly_accepted':probe.returncode==0}
save(); print('frozen/bundles done', flush=True)
canon = np.loadtxt(ROOT/'artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt',dtype=np.int64)
R['fulltests'] = {}
for name,tag,refpath in [('B03','20261113_231912',None),('B04','20261114_204223',None),
                        ('Q02clean','20261114_214407','artifacts/research/Q02_20261114/csim_q02_fulltest/csim_logits.txt'),
                        ('Q02replay','20261114_214206','artifacts/research/Q02_20261114/replay_cmodel_rows.txt')]:
    d=ROOT/('artifacts/results/raw_core_board_fulltest_'+tag)
    rr=rows(d/'fulltest.out'); ref=canon if refpath is None else np.loadtxt(ROOT/refpath,dtype=np.int64)
    ix=[int(r['idx']) for r in rr]
    b=np.array([[*map(int,r['logits'].split(',')),int(r['cls'])] for r in rr])
    extra={}
    if name.startswith('Q02'):
        regs=np.array([[int(r[k]) for k in ['quality_status','quality_detail','logit_margin']] for r in rr])
        extra={'bad_register_rows':int((regs!=ref[:,6:9]).any(1).sum()),'input_flagged':int(((regs[:,0]&7)!=0).sum())}
    else: extra={'integrity_nonzero':sum(int(r['integrity'])!=0 for r in rr)}
    meta=load(str(d.relative_to(ROOT)/'run_meta.json')) if (d/'run_meta.json').exists() else None
    R['fulltests'][name]={'rows':len(rr),'ordered_complete':ix==list(range(len(ref))),
                          'bad_logit_class_rows':int((b!=ref[:,1:6]).any(1).sum()),**extra,'run_meta':meta}
    if name=='Q02clean': R['fulltests'][name]['reference_equal_canonical']=bool((ref[:,1:6]==canon[:,1:6]).all())
    if name=='Q02replay':
        y=np.loadtxt(ROOT/'artifacts/research/Q02_20261114/replay_labels.txt',dtype=np.int64)
        R['replay_accuracy_audit']={'reported_accuracy':load(str(d.relative_to(ROOT)/'fulltest_check.json'))['accuracy'],
                                    'correct_accuracy_replay_labels':float((b[:,4]==y).mean()),'windows':len(y)}
R['deployment_identity']={}
for name,f in R['fulltests'].items():
    m=f['run_meta']
    if m:
        p=ROOT/'artifacts/deploy'/m['deployment'].split('/')[-1]/'overlay/design_1_wrapper.bit'
        R['deployment_identity'][name]=sha(p)==m['board_bitstream_sha256']
R['rt']={}
for name,tag in [('B03','20261113_235257'),('B04','20261114_211153'),('Q02','20261114_221435')]:
    d=ROOT/('artifacts/results/raw_core_board_rt_'+tag)
    s=load(str(d.relative_to(ROOT)/'out/summary.json')); z=np.load(d/'out/rt_raw.npz')
    w=z['window_index']; response=1000*(z['done']-z['release']); scored=response[s['warmup']:]
    R['rt'][name]={'executed':len(w),'scored':len(scored),'misses':int((scored>s['period_ms']).sum()),
                    'max_ms':float(scored.max()),'bad_logits':int((z['logits']!=canon[w,1:5]).any(1).sum()),
                    'bad_class':int((z['cls']!=canon[w,5]).sum())}
save(); print('fulltest/rt done',flush=True)
R['injections']={}
for sub in ['b03_manifest','heldout','recovery']:
    d=ROOT/'artifacts/research/B04_20261114'/sub
    man=np.loadtxt(d/'manifest.txt',dtype=np.int64)
    rr=rows(d/'board_rows.txt'); ref=np.loadtxt(d/'cmodel_rows_lp64.txt',dtype=np.int64)
    b=np.array([[*map(int,r['logits'].split(',')),int(r['cls']),int(r['integrity'])] for r in rr])
    idx=man[:,0]; inj=man[:,1]!=0; changed=b[:,4]!=canon[idx,5]; flagged=b[:,5]!=0
    R['injections'][sub]={'rows':len(rr),'reference_bad_rows':int((b!=ref[:,1:7]).any(1).sum()),
        'manifest_rows_exact':all(int(r['row'])==i and [int(r[k]) for k in ['idx','site','word','bit']]==list(man[i]) for i,r in enumerate(rr)) and len(rr)==len(man),
        'class_changes':int((inj&changed).sum()),'silent':int((inj&changed&~flagged).sum()),
        'injected_flagged':int((inj&flagged).sum()), 'clean_rows':int((~inj).sum()),
        'clean_bad_class':int((~inj&changed).sum()),'clean_flagged':int((~inj&flagged).sum()),
        'cmodel_meta_exists':(d/'cmodel_meta.json').exists()}
save(); print('injections done',flush=True)
from tools.research import temporal_policies_v2 as t
streams=t.load_streams(); saved=load('artifacts/research/T01v2_20261114/summary.json')
R['T01']={}
for sp,(cls,lg,y,rec) in streams.items():
    man=load('artifacts/research/T01v2_20261114/event_manifest_'+sp+'.json')
    t.check_manifest(man,rec,y)
    digest=hashlib.sha256(json.dumps(man['events'],sort_keys=True).encode()).hexdigest()
    checks={}
    for k in ['raw','vote_5','hyst_3','evid_64','evidto_64_4']:
        src=saved['events'].get(sp,{}) if sp=='validation' else {k:v['events'] for k,v in saved['test_report_selected_previously_explored'][sp].items()}
        if k not in src: continue
        got=t.summarize_events(t.run_events(man,cls,lg,*t.parse_key(k)),man)
        checks[k]=got==src[k]
    R['T01'][sp]={'manifest_labels_and_boundaries_valid':True,'digest_matches':digest==man['sha256_events'], 'events_recomputed_equal':checks}
R['T01_gate']=saved['selection_validation']
save(); print('T01 done',flush=True)
from tools.research import q01_quality_v2 as q
s=load('artifacts/research/Q01v2_20261114/summary.json')
R['recovery_counterexample']=s['bursts']['test']['flags']['lag32/d20']
R['Q01']={'sample_manifest_matches':q.sample_manifest()=={k:v for k,v in load('artifacts/research/Q01v2_20261114/sample_manifest.json').items() if k in ['val','test']},
    'confidence_example':q.confidences(np.array([[20,10,0,-10]]))[0].tolist(),'gates':s['gate_Q01_on_validation']}
with gzip.open(ROOT/'artifacts/research/Q02_20261114/q02_burst_replay_test.i32.gz','rb') as f:
    wins=np.frombuffer(f.read(),dtype='<i4').reshape(-1,1024,8)
replay=rows(ROOT/'artifacts/results/raw_core_board_fulltest_20261114_214206/fulltest.out')
lg=np.array([list(map(int,r['logits'].split(','))) for r in replay])
from tools.research.q02_check import reference
st,de,mg=reference(wins,lg)
R['Q02_independent_python_quality_vs_board']={'windows':len(wins),'bad_status':sum(int(r['quality_status'])!=x for r,x in zip(replay,st)),
    'bad_detail':sum(int(r['quality_detail'])!=x for r,x in zip(replay,de)), 'bad_margin':sum(int(r['logit_margin'])!=x for r,x in zip(replay,mg))}
save(); print('Q01/Q02 replay done',flush=True)
# Re-hash the complete off-repository copy, including large binaries/replay inputs.
idx=load('artifacts/research/Z01_20261114/archive_index.json'); av=load('artifacts/research/Z01_20261114/archive_verify.json')
dest=Path(av['dest']); bad=[]; current_bad=[]
for i,e in enumerate(idx['entries']):
    p=dest/'files'/e['path']
    if not p.is_file() or sha(p)!=e['sha256']: bad.append(e['path'])
    src=ROOT/e['path']
    if not src.is_file() or sha(src)!=e['sha256']: current_bad.append(e['path'])
    if i%250==0: print('archive',i,flush=True)
bv=subprocess.run(['git','bundle','verify',str(dest/'repository.bundle')],cwd=ROOT,capture_output=True,text=True)
R['archive']={'files':len(idx['entries']),'bytes':idx['bytes'],'bad_backup':bad,'current_changed_since_snapshot':current_bad,
    'bundle_hash_matches':sha(dest/'repository.bundle')==av['git_bundle_sha256'],'bundle_verify_rc':bv.returncode,
    'same_physical_disk_as_repo':av['same_physical_disk_as_repo']}
save(); print('DONE',flush=True)
