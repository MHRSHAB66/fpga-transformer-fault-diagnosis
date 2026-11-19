"""Read-only numerical pilots. Run from repository root; never accesses board."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import re
import numpy as np
from src.quantize import integer_model as im

out_dir = Path(__file__).resolve().parent
header = Path('hls/f1_int10_seq256/vitis/weights_int10.h')
source = Path('src/quantize/integer_model.py')
s = header.read_text()
arrays = {m.group(1): [int(x) for x in m.group(2).split(',') if x.strip()]
          for m in re.finditer(r'static const (?:long long|long|int)\s+(\w+)\[[^\]]+\]\s*=\s*\{([^}]+)\};', s)}
layers = []
for name, din in [('proj',8),('l0_wq',32),('l0_wk',32),('l0_wv',32),
                  ('l0_out',32),('l0_ff0',32),('l0_ff3',64),('cls1',32),('cls4',16)]:
    weights, biases = arrays['W_'+name], arrays['b_'+name]
    assert len(weights) == len(biases)*din
    bound = max(511*sum(abs(x) for x in weights[o*din:(o+1)*din])+abs(b)
                for o,b in enumerate(biases))
    multiplier = arrays['M_'+name+'_q40'][0]
    assert multiplier > 0
    threshold = (512*(1<<40)+multiplier-1)//multiplier
    layers.append(dict(layer=name, bound_including_bias=bound,
                       sufficient_signed_accumulator_bits=bound.bit_length()+1,
                       multiplier_q40=multiplier,
                       unsigned_multiplier_bits=multiplier.bit_length(),
                       sufficient_signed_product_bits=(bound*multiplier).bit_length()+1,
                       conservative_saturation_threshold=threshold,
                       sufficient_signed_unsaturated_input_bits=min(bound,threshold).bit_length()+1))

dummy = SimpleNamespace(_cur_layer=0, exp_lut={0:(np.array([32768],dtype=np.int64),0)})
rows=[]
original_bits = im.ATTN_BITS
try:
    for tokens in (128,256,512,1024):
        for bits in (8,9,10):
            im.ATTN_BITS=bits
            scores=np.zeros((1,1,tokens,tokens),dtype=np.int64)
            attention=im.IntegerTransformer._softmax(dummy,scores,None)
            rows.append(dict(tokens=tokens,score_shape=list(scores.shape),signed_attention_bits=bits,
                             first_code=int(attention[0,0,0,0]),
                             first_row_sum=int(attention[0,0,0].sum()),
                             zero_rows=int((attention.sum(axis=-1)==0).sum())))
finally:
    im.ATTN_BITS=original_bits

result=dict(review_date='2026-11-11',board_access=False,training_or_dataset_access=False,
            actual_reproduction="python -c \"import runpy; from pathlib import Path; runpy.run_path(str(next(Path('research_notes').glob('*/hardware_pilot.py'))), run_name='__main__')\"",
            assumptions=['All internal linear inputs are clamped to [-511,511] by approved reference; ap_int<10> bit pattern -512 is not included in this legal-domain proof.',
                         'Weights/bias/scales are the exact checked-in header constants, not noiseaug exports.',
                         'Row softmax dummy has exp(0)=32768, all score entries=0, no mask and no trained model.',
                         'No resource/timing/accuracy claims follow from these pilots.'],
            hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (header,source)},
            layer_bounds=layers,actual_softmax_equal_score_pilot=rows)
(out_dir/'hardware_pilot.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Saved research evidence: hardware_pilot.json')
