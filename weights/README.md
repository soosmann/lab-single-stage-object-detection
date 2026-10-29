# Local checkpoints

The notebooks save PyTorch state dictionaries here:

- `yolo_detector.pt`: initial detector run.
- `yolo_detector_no_filter.pt`: summed cross-entropy experiment.
- `yolo_detector_focal.pt`: summed focal-loss experiment.

Checkpoints are ignored by Git. The existing local `yolo_detector.pt` has been
preserved; a fresh clone requires training or a separately obtained checkpoint.
No download location is supplied for these student-trained weights.

Load a compatible state dictionary without downloading backbone weights:

```python
from pathlib import Path
import torch
from single_stage_detector import SingleStageDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SingleStageDetector(pretrained=False).to(device)
state = torch.load(Path('weights/yolo_detector.pt'), map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
```

The prediction head keeps the original `pred_network.net.*` parameter names.
Anchors move with the model and do not add an extra checkpoint key.
