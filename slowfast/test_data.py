
from datasets import hmdb51
from config.defaults import get_cfg
cfg = get_cfg()
# Load config from cfg.
cfg.merge_from_file('../configs/HMDB/I3D.yaml')

data = hmdb51.Hmdb51(cfg, 'train')

sam = data[6]
labels = [data.data[i]['label'] for i in range(len(data.data))]
print(min(labels))
print(max(labels))
