# Attribution and reuse

This repository contains university coursework and inherited teaching utilities.

- **Student implementation and report:** Marcel Hofmann and Carlos Celaya.
- **UAM lab material:** Pablo Carballeira, Escuela Politécnica Superior,
  Universidad Autónoma de Madrid. The notebook's original introduction credits
  him and identifies the University of Michigan material below as a source.
- **Adapted teaching material and helpers:** Justin Johnson, Electrical
  Engineering and Computer Science, University of Michigan. The `eecs598`
  package and assignment helper conventions were supplied with the lab.
- **Dataset:** PASCAL Visual Object Classes 2007. Images and annotations are
  external data, retain their original terms, and are not included in Git.
- **Evaluation:** [Cartucho/mAP](https://github.com/Cartucho/mAP), fetched
  separately when running the optional evaluation cells. Its own license applies.
- **Model and framework:** PyTorch and torchvision, including MobileNetV2
  ImageNet weights downloaded through torchvision.

The detector is inspired by YOLO and uses anchor-based object detection concepts.
The original notebook retains the paper references and teaching explanations.

The repository cleanup reorganizes the original files, extracts the completed
detector into `src/`, documents the experiments, and fixes portability issues.
The supporting `eecs598` utilities are retained to preserve the lab imports;
some utilities belong to other assignments and are not part of this detector.
The two small visual examples in `docs/assets/` were extracted from the original
notebook outputs and contain VOC imagery.

No license file was supplied with the lab. A blanket open-source license has not
been added because ownership and redistribution terms for the inherited course
material have not been established. This document records provenance and does
not grant additional reuse rights.
