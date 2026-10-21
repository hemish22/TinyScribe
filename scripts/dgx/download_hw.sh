#!/usr/bin/env bash
# Handwriting word datasets for the handwritten pages.
#   bash scripts/dgx/download_hw.sh
# tools/hw_datasets/ holds one folder per source. EMNIST (Latin glyphs) and IIIT Hindi come from the
# IndicFormBench checkout (symlinked); IIIT-INDIC-HW-WORDS Tamil (CVIT, IIIT Hyderabad; research use)
# comes from its Hugging Face parquet mirror, about 5.4 GB. BanglaWriting (Mridha et al. 2020,
# Mendeley Data 10.17632/r43wkvdk4w.1, CC BY-NC 3.0; 260 writers, word boxes + labels) gives the
# Bengali words, 132 MB.
set -euo pipefail
cd "$(dirname "$0")/../.."
HW=tools/hw_datasets
IFB_HW="$HOME/indicformbench/tools/hw_datasets"
if [ -L "$HW" ]; then mv "$HW" "$HW.indicformbench-link"; fi
mkdir -p "$HW"
for d in emnist iiit_hw_hindi; do [ -e "$HW/$d" ] || ln -s "$IFB_HW/$d" "$HW/$d"; done
.venv/bin/hf download c3rl/IIIT-INDIC-HW-WORDS-Tamil --repo-type dataset --local-dir "$HW/iiit_hw_tamil"
du -sh "$HW/iiit_hw_tamil"

BW="$HW/banglawriting"; mkdir -p "$BW"
if [ ! -d "$BW/converted" ]; then
  curl -sL -o "$BW/converted.zip" \
    "https://data.mendeley.com/public-files/datasets/r43wkvdk4w/files/070a3d6f-675e-4e29-9f4d-77430e8950d1/file_downloaded"
  echo "263ca3f31a4f0c3a197194f7e0c35b383fb7ef5f96942b08d96d9272fb9f9366  $BW/converted.zip" | sha256sum -c
  .venv/bin/python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$BW/converted.zip" "$BW/converted"
fi
du -sh "$BW"
