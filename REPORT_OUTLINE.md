# Report outline (3000–4000 words) – aligned to the marking rubric

Target total: ~3,600 words. Rubric weights in brackets.

| # | Section | Words | Rubric criterion | What to write |
|---|---|---|---|---|
| 1 | Introduction and problem definition | 450 | Problem & justification (20%) | Dubai is multilingual; reviews arrive in Arabic, English and mixed text; managers cannot read thousands. Who is affected, cost of the problem, why existing English-only tools fail. |
| 2 | Why NLP/LLMs? | 300 | Problem & justification (20%) | Why rules or manual reading fail; what each NLP task contributes (classification, clustering, summarisation, retrieval). Cite 4–6 papers (AraBERT, XLM-R, CAMeLBERT, RAG, BERTopic). |
| 3 | Data | 550 | Data & pre-processing (15%) | Datasets used, size, label balance, language mix, licences, the synthetic code-switched augmentation and its limits. Figure: `results/eda.png`. |
| 4 | Pre-processing and embeddings | 450 | Data & pre-processing (15%) | Arabic normalisation (diacritics, tatweel, alef/ya/ta marbuta, elongation), PII masking, light vs aggressive cleaning and why, language detection, clause splitting; TF-IDF vs subword vs sentence embeddings. |
| 5 | System architecture | 600 | Pipeline design (25%) | Pipeline diagram (README), each stage's input/output, why XLM-R, why lexicon-guided aspects, why k-means + c-TF-IDF, why evidence-pack prompting. Alternatives considered. |
| 6 | Model selection, training and tuning | 350 | Pipeline design (25%) | Baseline vs zero-shot vs fine-tuned, hyper-parameters (lr 2e-5, 3 epochs, batch 32, max_len 128, class weights), validation protocol, seed. |
| 7 | Tools and platforms | 200 | Tools (10%) | Hugging Face Transformers, sentence-transformers, scikit-learn, FAISS, Gradio, Colab, GitHub – role of each in one sentence. |
| 8 | Evaluation and results | 400 | Pipeline design / Clarity | Table `results/model_comparison.csv`, per-language F1, confusion matrix, error analysis themes (sarcasm, dialect, neutral class), summary faithfulness. |
| 9 | Ethical and responsible AI | 450 | Ethics (15%) | Bias across languages/dialects, privacy (PII masking), fake reviews, hallucination, over-reliance and staff fairness, transparency. Each with a concrete mitigation implemented or proposed. |
| 10 | Expected output and value proposition | 200 | Problem & justification | What a manager receives, time saved, who benefits, deployment ideas. |
| 11 | Limitations and future work | 150 | Clarity | Keyword aspects, non-Dubai data, dialect coverage; next: real Dubai data, aspect NLI, human evaluation. |
| 12 | Conclusion + references | 100 | Clarity (15%) | Short wrap-up; consistent referencing style (Harvard/APA as the unit requires). |

Checklist before submitting
- [ ] Word count 3000–4000 (excluding references)
- [ ] Every figure/table is referred to in the text
- [ ] Numbers in the report match `results/` files
- [ ] GitHub link works in a private browser window; README renders
- [ ] You can explain every component without notes (demo questions)
