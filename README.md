# LLM Product Attribute Tagging: Gemini vs a Classic ML Baseline

Can a large language model tag fashion products (gender, article type, colour, usage) from the title alone,
and when is it better than a model trained on labelled data? Tested on Myntra catalogue data with known labels.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaishvin/llm-product-tagging/blob/main/notebooks/llm_product_tagging.ipynb)

## Business problem
Every new listing needs attributes that drive search filters, category reporting and demand planning. Tagging
by hand is slow, and a trained classifier needs thousands of labelled examples, which a new category or
marketplace doesn't have. LLMs promise tagging with no training data. This project measures whether that holds up.

## Approach
| Step | What | Tools |
|---|---|---|
| 1. Data | 44k Myntra products with catalogue tags; top 20 article types; balanced test set of 300 titles (15 per type) | pandas |
| 2. Baseline | TF-IDF (word 1–2 grams) + logistic regression, one model per attribute | scikit-learn |
| 3. LLM | Gemini zero-shot and 12-example few-shot prompts, 25 titles per call, **JSON output constrained by a response schema** to the allowed label values | Gemini API |
| 4. Evaluate | Accuracy and macro-F1 per attribute, "all 4 correct" rate, token use and latency | scikit-learn |
| 5. Data needs | Retrain the baseline on 25 → all labelled titles to find where it catches up with the LLM | scikit-learn |
| 6. Errors | Confusion matrix and error review: model mistakes vs ambiguous catalogue labels | pandas, matplotlib |

Engineering details: batching to stay within free-tier rate limits, retries with backoff on 429 errors, cached
raw responses (re-runs don't spend quota), and validation of every returned value against the allowed list.

## Results
<!-- Fill in after running the notebook -->
| Method | Labelled data used | gender | articleType | baseColour | usage | All 4 correct |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | __ titles | __% | __% | __% | __% | __% |
| Gemini zero-shot | 0 | __% | __% | __% | __% | __% |
| Gemini few-shot | 12 | __% | __% | __% | __% | __% |

![Accuracy by method](results/accuracy_by_method.png)
![Learning curve](results/learning_curve.png)

**Key findings**
- _Which method wins on which attribute_
- _How much labelled data the baseline needed to match the LLM_
- _What the errors show (model mistakes vs noisy labels)_

## Run it
1. Click **Open in Colab** above.
2. Add two Colab secrets: `KAGGLE_API_TOKEN` (Kaggle → Settings → API Tokens) and `GOOGLE_API_KEY`
   (free key from [Google AI Studio](https://aistudio.google.com/apikey)).
3. Runtime → Run all (about 5 minutes; ~26 API calls on the free tier).

Set `DEMO_MODE = True` to test the notebook with synthetic data and a fake LLM, no keys needed.

## Repo structure
```
notebooks/llm_product_tagging.ipynb   main notebook (Colab-ready)
notebooks/llm_product_tagging.py      same notebook as a plain script
results/                              metrics, charts, error analysis, raw LLM responses
requirements.txt
```

## Data
[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
by Param Aggarwal on Kaggle (Myntra catalogue metadata). Only `styles.csv` is used; data is not included in this repo.
