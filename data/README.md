# Local PASCAL VOC data

The notebooks use this directory as the dataset root. On first use,
`get_pascal_voc2007_data(DATA_ROOT, 'train')` downloads and extracts the VOC 2007
trainval archive. Validation uses the `val` split of the same archive.

If you already have the dataset, arrange it as follows:

```text
data/
└── VOCdevkit/
    └── VOC2007/
        ├── Annotations/
        ├── ImageSets/
        └── JPEGImages/
```

An existing `VOCtrainval_06-Nov-2007.tar` can also be placed here for torchvision
to reuse. If the original download host is unavailable, the notebooks' original
teaching material lists this [University of Michigan mirror](https://web.eecs.umich.edu/~justincj/data/VOCtrainval_06-Nov-2007.tar).
Extract the archive here, preserving its `VOCdevkit/VOC2007` hierarchy.

Reserve about 1 GB for the archive and extracted images. Dataset files and the
old `extracted.txt` marker are ignored by Git; extraction is detected from the
dataset directory itself. The data retains its original dataset terms.
