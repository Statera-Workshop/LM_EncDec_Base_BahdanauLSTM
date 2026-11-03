# LM_EncDec_Base_BahdanauLSTM

Encoder-decoder translation model ported from legacy TF2 code C4 to PyTorch (cell=lstm, attention=bahdanau).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 32.9 M
- Test perplexity 35.5, next-word accuracy 0.556
- BLEU-4 greedy (whole test split): 18.8
- BLEU-4 beam width 4 on 500 pairs: 20.9 (greedy on the same pairs 20.1)
- Examples:
  - `this winter has been mild .` -> `cette viande est arrive les faits .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils vont ici , trois heures .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `je me suis vecu dans le parc .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_bahdanau_lstm
```
