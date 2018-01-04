# Keyword Spotter

Spots keywords with a nearest neighbour like approach and dynamic time warping as metric.

## Prerequisites

* Python >= 3.6
* scipy
* PIL
* fastdtw


## Usage

```bash
python3 main.py
```

Note that a lot of Keywords only occur once, so precision and recall will can be 0.

To run the spotter on a single word image and get a list of images that are similar to it, run:

```bash
python3 keyword_spotter.py ground-truth/word-images_preprocessed/271-34-08.png
```

## Notes

Preprocessing was done with `extract_word_images.py` (for cropping) and
`preprocess_word_images.py` (for binarization).