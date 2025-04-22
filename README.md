# Waste Recycling Classification

[![Weights | available](https://img.shields.io/badge/Weights-available-red)](https://drive.google.com/drive/folders/1cLbZhj7afSI0w5M-bXx25hB88afFwKUo)

## Setup

1. Clone or download this repository.
2. Install package dependencies (e.g. using `pip install -r requirements.txt`).
3. Optionally, download our weights to a directory named `models` in the project's root.
4. Run `main.ipynb` cell-sequentially, as desired!

## Notes

1. Some operating systems may be sensitive to the capitalisation of the `Dataset` directory: one may rename it to `dataset` if an error occurs due to such a convention.
2. Assuming the first three steps were completed, the project's root should resemble the following.

```
./
└── Dataset/
    └── non_recyclable/
    └── recyclable/
└── models/
    └── IV4/
    └── MNV2/
    └── RN152/
    └── RNRS50/
    └── SMNV2/
└── architectures.py
└── main.ipynb
└── utils.py
```
