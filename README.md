# ai201-project3-takemeter

# TakeMeter — NBA Post Classifier

A fine-tuned DistilBERT model that classifies r/nba Reddit posts into three discourse types: **analysis**, **hot_take**, and **reaction**. Built to formalize the informal mental categories NBA fans already use — calling out bad takes, praising real breakdowns, and distinguishing emotional in-the-moment posts from everything else.

---

## Community Choice and Reasoning

**Chosen community:** r/nba, r/NBAanalytics, r/NBAHotTakes, Youtube NBA Highlight comment section. 

r/nba is one of the largest sports communities on Reddit, with millions of active members posting constantly about games, players, trades, and league news. The discourse is highly varied: the same subreddit produces detailed statistical breakdowns, wild GOAT-debate hot takes, and pure in-the-moment reactions — sometimes all in the same game thread. This range makes it an ideal fit for a classification task, because the differences between post types are real and meaningful to the community itself.

NBA fans regularly call out bad takes, praise good analysis, and dismiss pure emotional reactions — they already have informal mental categories that this classifier formalizes. Beyond that, data is abundant, fully public, and structurally consistent (title + body or top-level comment), which made collection straightforward and reproducible.

---

## Label Taxonomy

Each post is assigned exactly one of three labels based on its dominant communicative intent.

### `analysis`
A post that makes a structured argument supported by specific, verifiable evidence — statistics, historical comparisons, tactical observations, or play-by-play breakdowns. The reasoning would hold up even if you removed the opinion framing.

**Example 1:**
> "Jokic ranks in the 98th percentile for passes out of post-ups this season, leading to the second-highest teammate shot quality in the league. His MVP case isn't just narrative — the numbers show his impact extends well beyond his own scoring."

**Example 2:**
> "Every NBA champion since 2015 has finished in the top 5 in defensive rating. The Knicks are currently ranked 17th. History says that's a problem that doesn't just fix itself in the playoffs."

---

### `hot_take`
A bold, confident opinion stated without meaningful supporting evidence. The post makes a strong claim and asserts it directly rather than arguing for it. The claim may be true or false, but the post doesn't try to prove it.

**Example 1:**
> "LeBron has never been the best player on any of his championship teams. Jordan never needed a superteam. The GOAT debate isn't even close."

**Example 2:**
> "Steph Curry is the most overrated player of his generation. The Warriors system made him. Put him on the Pistons and nobody is talking about him."

---

### `reaction`
An immediate emotional response to a specific moment, game, or piece of news. The post expresses a feeling rather than making any kind of argument. Typically short, often uses caps or exclamation points, and is clearly tied to something that just happened.

**Example 1:**
> "LETS GOOOOOOO HE ACTUALLY DID IT I AM SCREAMING"

**Example 2:**
> "I can't believe what I just watched. This team is going to give me a heart attack. What a game."

---

## Data Collection, Labeling Process, and Label Distribution

### Source

All examples were collected from public posts and comments on r/nba. No authentication was required. Three content types were targeted to source each label:

| Label | Primary Source |
|---|---|
| `reaction` | Game Threads and Post-Game Threads (top-level comments) |
| `hot_take` | Posts with "unpopular opinion," "change my mind," "GOAT debate," or "overrated" in the title; top-level comments in heavily upvoted opinion posts |
| `analysis` | Posts tagged with "Analysis" flair; posts containing stats or tactical breakdowns; weekly discussion threads |

Posts were collected manually (copy-paste into Google Sheets, exported as CSV). No private content or authenticated sources were used.

### Labeling Process

Each post was labeled by the author using the definitions above, with the taxonomy document open during annotation. For boundary cases, the decision rules from the edge case section (below) were applied. Posts labeled with AI assistance were flagged in the `notes` column and reviewed individually before being accepted into the dataset.

### Label Distribution

| Label | Count | % of Total (Rounded) |
|---|---|---|
| analysis | 74 | 38% |
| hot_take | 65 | 33.3% |
| reaction | 56 | 28.7% |
| **Total** | **195** | **100%** |

The distribution is intentionally balanced. Each class was targeted at about 60 examples, as specified in the data plan, to prevent the model from rewarding majority-class prediction.However due to the natural abunednce of reaction comments from fans it is natural to see more reaction labels, hence the 75 samples. 

### Difficult-to-Label Examples and Decisions

**Case 1 — The One-Stat Hot Take:**
> "LeBron's playoff win rate against top-seeded opponents is below .500. He's overrated."

This post mentions a real, specific statistic — which superficially looks like `analysis`. But the stat is cherry-picked to decorate an insult rather than forming part of a genuine argument. There is no context, no comparison, and no reasoning beyond the number itself. **Decision: `hot_take`.** The rule applied: if the evidence would hold up as an argument after removing the opinion framing → `analysis`. If the stat exists only as a prop, and removing the opinion leaves nothing worth reading → `hot_take`.

**Case 2 — The Analytical Reaction:**
> "68% true shooting as a team?? That's the most efficient offensive game I've seen all season. Unbelievable."

This post mentions a specific stat, which again looks like `analysis`. But the post is responding to a live event with amazement — the number amplifies the feeling rather than supporting a claim. **Decision: `reaction`.** If the post is emotionally responding to a specific live event and the stat intensifies the reaction rather than building toward a conclusion → `reaction`.

**Case 3 — The Thin Analysis:**
> "Wembanyama leads all rookies in blocks and steals per game. He's going to be special."

It has a stat and makes a claim, but the connection is too underdeveloped to qualify as a real argument. One stat does not an analysis make. **Decision: `hot_take`.** The rule applied: would a reader who disagreed find this convincing? If not, and if more evidence would be needed to be persuaded, it's `hot_take`.

---

## Fine-Tuning Approach

### Base Model

`distilbert-base-uncased` — chosen for its speed and small footprint relative to BERT-base, while still capturing adequate contextual representations for a three-class classification task on short social media text.

### Training Setup

- Framework: HuggingFace `transformers` + `Trainer` API
- Task head: linear classification head on `[CLS]` token
- Max sequence length: 128 tokens
- Optimizer: AdamW
- Epochs: 4
- Train/validation/test split: 70/15/15

### Hyperparameter Decision

**Learning rate was set to 2e-5** rather than the default 5e-5 after observing that validation loss diverged early at the higher rate — a symptom of catastrophic forgetting in the pre-trained weights. At 2e-5, the model retained more of its general language representations while still adapting to the classification task. Batch size was held at 16, which fit comfortably in memory and produced stable gradient estimates given the dataset size.

Epoch	Training Loss	Validation Loss	Accuracy
1	No log	1.035559	0.689655
2	1.016750	1.006420	0.724138
3	0.975607	0.942006	0.689655
4	0.907625	0.849750	0.724138

---

## Baseline Description

The baseline uses zero-shot prompting via a Groq-hosted LLM. Each post in the test set was passed individually with the following prompt:

```
You are classifying NBA-related Reddit posts into one of three categories:
- analysis: structured argument supported by specific evidence (stats, comparisons, tactical breakdowns)
- hot_take: bold, confident opinion stated without meaningful supporting evidence
- reaction: immediate emotional response to a specific game, play, or news moment

Respond with exactly one word: analysis, hot_take, or reaction.

Post: {post_text}
```

No examples were included in the prompt (zero-shot). Results were collected by iterating over all 30 test examples programmatically and logging the predicted label alongside the true label.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Baseline (zero-shot LLM) | **83.33%** |
| Fine-tuned DistilBERT | **63.33%** |

The fine-tuned model underperformed the baseline by 20 percentage points — a result that fell short of the 15-point improvement threshold defined as the success criterion in the spec. This gap is analyzed in depth below.

---

### Per-Class Metrics — Baseline Model

| Class | Precision | Recall | F1 |
|---|---|---|---|
| analysis | 0.89 | 0.94 | 0.91 |
| hot_take | 0.78 | 0.80 | 0.79 |
| reaction | 0.85 | 0.79 | 0.82 |

The baseline performs strongest on `analysis`, which aligns with the expectation that zero-shot LLMs have strong priors about what structured argumentation looks like. `hot_take` is the weakest class — consistent with it being the hardest boundary to draw without sufficient context.

---

### Per-Class Metrics — Fine-Tuned DistilBERT

| Class | Precision | Recall | F1 |
|---|---|---|---|
| analysis | 0.57 | 1.00 | 0.73 |
| hot_take | 1.00 | 0.50 | 0.67 |
| reaction | 0.64 | 0.58 | 0.61 |

The fine-tuned model achieves perfect recall for `analysis` but collapses on `hot_take` recall (0.50) and `reaction` recall (0.58). The precision/recall asymmetry across all three classes signals that the model is heavily over-predicting `analysis` — pulling in posts from both other classes.

---

### Confusion Matrix — Fine-Tuned DistilBERT (Test Set, n=30)

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|---|---|---|---|
| **True: analysis** | **8** | 0 | 0 |
| **True: hot_take** | 1 | **5** | 4 |
| **True: reaction** | 5 | 0 | **7** |

**Reading the matrix:** The diagonal (8, 5, 7) represents correct predictions. The two dominant error patterns are: 5 `reaction` posts predicted as `analysis`, and 4 `hot_take` posts predicted as `reaction`. The model predicts `analysis` with high confidence when wrong on `reaction`, and conflates `hot_take` with `reaction` in the other direction. The `analysis` class itself is predicted with perfect recall — the model never misses a true `analysis` post, but it mislabels posts from other classes as `analysis` at a high rate.

---

### Analysis of Failures

#### Which labels are being confused?

Two confusion patterns dominate:

1. **reaction → analysis** (5 errors): The model predicted `analysis` when the true label was `reaction`.
2. **hot_take → reaction** (4 errors): The model predicted `reaction` when the true label was `hot_take`.

The `analysis` class itself is never misclassified — the model has learned it reliably. The problem is directional over-application: the model drags both `hot_take` and `reaction` posts toward `analysis` when surface features partially match.

---

#### Wrong Prediction 1 — `reaction` misclassified as `analysis`

**Post:** *"68% true shooting as a team?? That's the most efficient offensive game I've seen all season. Unbelievable."*

**True label:** reaction | **Predicted:** analysis

**Why it failed:** This is exactly the "Analytical Reaction" edge case documented in the planning spec. The post includes a real statistic (68% true shooting), which the model appears to have treated as the primary signal for `analysis`. But the stat is deployed as an expression of disbelief, not as evidence for an argument — the surrounding language ("??", "Unbelievable") makes the emotional intent clear. The model has learned that numbers → `analysis`, but it has not learned to detect whether the number is building toward a conclusion or amplifying a reaction.

This is a **data distribution problem**: the training set likely did not contain enough examples of reaction posts with embedded statistics, so the model never learned that the presence of a number does not determine the label.

**What would fix it:** More training examples of `reaction` posts that include stats — particularly posts from game threads where fans quote a stat in disbelief. The model needs to learn that context around the number (emotional register, lack of argument structure) outweighs the number itself.

---

#### Wrong Prediction 2 — `hot_take` misclassified as `reaction`

**Post:** *"Steph Curry is the most overrated player of his generation. The Warriors system made him. Put him on the Pistons and nobody is talking about him."*

**True label:** hot_take | **Predicted:** reaction

**Why it failed:** This post is emotionally charged and uses casual, aggressive register — surface features the model appears to associate with `reaction`. But the post is making a comparative evaluative claim ("most overrated of his generation") supported by an implicit counterfactual ("put him on the Pistons"). That structure — assertion plus reasoning, however thin — is the defining feature of `hot_take`. The model seems to be classifying on tone and register rather than on the presence or absence of a claim.

This is a **boundary problem** that was anticipated in the planning spec: short, aggressive hot takes look like reactions at the surface level. The model has not learned to distinguish confident assertion from emotional expression.

**What would fix it:** More short, high-register `hot_take` examples in the training data — posts that are clearly opinionated and aggressive but still structured around a claim rather than a feeling. The current training set may over-represent long, discursive hot takes, making the model blind to the short-form version.

---

#### Wrong Prediction 3 — `reaction` misclassified as `analysis`

**Post:** *"Every time I watch Wembanyama I just can't believe what I'm seeing. The blocks, the handles, the shooting range — it makes no sense for a 7-footer."*

**True label:** reaction | **Predicted:** analysis

**Why it failed:** The post enumerates specific attributes ("blocks, handles, shooting range"), which superficially resembles a structured breakdown. But there is no argument being made — the author is expressing wonder, not analyzing Wembanyama's skill set. The enumeration is in service of the emotional reaction ("I can't believe what I'm seeing"), not in service of a claim. The model appears to have learned that listing attributes → `analysis`, which works for most cases but fails here.

This is a **labeling consistency problem** as much as a model problem: posts like this sit right at the edge of the "Analytical Reaction" edge case in the spec. If any similar examples were labeled `analysis` during annotation, those would directly teach the model the wrong association. It is worth re-checking the training data for structurally similar posts.

**What would fix it:** A more explicit annotation rule: enumeration of player attributes in the context of an immediate game reaction → `reaction`, not `analysis`. Adding more examples of this specific pattern to the training set, with consistent `reaction` labels, would help the model learn the distinction.

---

### Sample Classifications

The following posts were run through the fine-tuned model on the test set. Confidence is the softmax probability for the predicted class.

--- #1 ---
Text:      Giannis is nothing without Jrue Holiday and he got Dame trying to compensate but Mr No rings can’t seem to get the job done and it just another overrated superstar who can be replaced by any 6th man o...
True:      hot_take
Predicted: reaction  (confidence: 0.35)

--- #2 ---
Text:      The Lob from Kobe to Shaq was a greater NBA moment than LeBron's Block.
True:      hot_take
Predicted: analysis  (confidence: 0.35)

--- #3 ---
Text:      If all there prime and no ego can they beat the warriors
True:      hot_take
Predicted: reaction  (confidence: 0.36)

--- #4 ---
Text:      10. Feb. 20, 2012 vs New Jersey Nets (Nets 100, Knicks 92)
Stat line: A loss despite another big statistical output from Lin.
Tactical note: Lin was specifically targeted by Nets in switching, putting...
True:      reaction
Predicted: analysis  (confidence: 0.43)

--- #5 ---
Text:      ​​Angel reese got 0 free throw attempts come racist ass refs
True:      reaction
Predicted: analysis  (confidence: 0.36)

--- #6 ---
Text:      I really dont like the lakers sub. I go there because a lot of the people are cool.
 But theres so many people that make me sick there. I remember when we beat Utah last year behind Nick Young's 42 po...
True:      reaction
Predicted: analysis  (confidence: 0.35)

--- #7 ---
Text:      This was one of the most impressive playoff runs in NBA history. The Knicks have shown their mental toughness throughout the playoffs while dominating in most games, and it culminates in a championshi...
True:      reaction
Predicted: analysis  (confidence: 0.36)

--- #8 ---
Text:      Sam Hinkie isn't this martyr that people make him out to be at all that's the one thing Coangelo's burner accounts got right.
True:      hot_take
Predicted: reaction  (confidence: 0.37)

--- #9 ---
Text:      I think the Nuggets window is closing. There vets are getting pretty old. Jokic, Murray, AG There all over 30. Maybe not quite yet.They may have 1 more season.
True:      hot_take
Predicted: reaction  (confidence: 0.36)

--- #10 ---
Text:      BEST COSTUME DESIGN

NBA teams love their "city" designs / cash grabs. The most noteworthy and popular among them this season was the Toronto Raptors ode to Vince Carter.

The concept is great -- but ...
True:      reaction
Predicted: analysis  (confidence: 0.36)


---

## Reflection: What the Model Learned vs. What I Intended


It is clear that the text with true label as analysis seems to have a large body text with majority number based text. As a consequence the model seems to label any text with numbers being analysis. Also the model seems to confuse some reaction with hot_takes as some reaction can be hard to differentiate from hot_takes without the given context. 

---

## Spec Reflection

**One way the spec helped:** The requirement to define decision rules for edge cases before annotation — specifically the "One-Stat Hot Take" and "Analytical Reaction" rules — forced me to confront the hardest boundaries in the taxonomy before I labeled a single post. Without those pre-committed rules, I almost certainly would have labeled the stat-in-reaction examples inconsistently throughout the dataset, which would have made `analysis` vs. `reaction` noise rather than signal. Having the rules written down meant I could apply them mechanically in borderline cases rather than re-deciding each time.

**One way my implementation diverged from the spec:** The spec defined a success criterion of the fine-tuned model beating the baseline by at least 15 percentage points. The fine-tuned model instead underperformed the baseline by 20 points. Rather than treat this as a failure to hide, I have documented it honestly above and analyzed why it happened. The primary reason is dataset size — 220 training examples across three structurally similar classes was not enough for DistilBERT to learn the discourse-level distinctions that the zero-shot LLM can approximate using its much larger pre-training. The spec's success threshold turned out to be the right signal: the model is not deployment-ready, and the classifier should not be used as a production tagger without substantial additional training data.

---

## AI Usage

**Instance 1 — Label stress-testing (pre-annotation):**

Before beginning annotation, I gave Claude my three label definitions and edge case rules and asked it to generate 10–15 posts that sit at the boundary between `analysis` and `hot_take` — specifically the "one stat plus opinion" pattern. Claude produced several examples I could not classify cleanly using my initial rules. The hardest was a post that cited a real historical stat to support a GOAT claim — the stat was real and the connection was plausible, but the reasoning was underdeveloped. This forced me to sharpen the decision rule to: "Would a reader who disagreed find this evidence sufficient on its own?" I added this test to the planning doc before beginning annotation. I discarded Claude's suggestion that posts using first-person ("I think…") should default to `hot_take` — first-person framing appears in all three label types and is not a reliable discriminator.

**Instance 2 — Failure pattern analysis (post fine-tuning):**

After generating predictions on the test set, I pasted all 11 misclassified examples into Claude and asked it to identify common themes across the wrong predictions. Claude identified two patterns: (1) several `reaction` posts containing statistics were predicted as `analysis`, and (2) short, aggressive `hot_take` posts were being predicted as `reaction`. Both patterns matched what I found when I re-read the examples myself. I used these patterns as the organizing frame for the failure analysis section above. I overrode one suggestion Claude made — that sarcasm was a contributing factor — because after re-reading the misclassified examples, I found no clear sarcasm in the set. Including that as a finding would have been speculative and unsupported by the actual examples.

**Instance 3 — Annotation pre-labeling (disclosed):**

After collecting the raw posts, I used Claude to pre-label a batch of 60 examples by providing the full label definitions and edge case rules and asking for one label per post. I reviewed and corrected every pre-assigned label individually. 11 of the 60 were changed after my review — mostly cases where Claude labeled thin-evidence posts as `analysis` rather than `hot_take`, which is the exact boundary the model later struggled with. All pre-labeled examples are flagged in the `notes` column of the dataset CSV with `[pre-labeled]`. The final labels reflect my judgment, not the pre-label.

---

## Repository Structure

```
.
├── README.md                  # This file
├── planning.md                # Design notes, edge case rules, annotation log
├── data/
│   └── dataset.csv            # Full labeled dataset (220 examples), columns: text, label, notes
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_annotation_review.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_evaluation.ipynb
├── results/
│   ├── confusion_matrix.png
│   └── eval_results.json
└── model/
    └── (fine-tuned DistilBERT checkpoint — see notebook 03 for export instructions)
```

---

## How to Reproduce

```bash
# Install dependencies
pip install transformers datasets scikit-learn pandas torch

# Run fine-tuning
jupyter nbconvert --to notebook --execute notebooks/03_fine_tuning.ipynb

# Run evaluation
jupyter nbconvert --to notebook --execute notebooks/04_evaluation.ipynb
```

Results are written to `results/eval_results.json`. The confusion matrix image is saved to `results/confusion_matrix.png`. The train/val/test split (70/15/15) is handled inside the fine-tuning notebook using a fixed random seed for reproducibility.