# TakeMeter — Planning Document
### AI201 · Project 3
---

## 1. Community

**Chosen community:** r/nba (Reddit)

r/nba is one of the largest sports communities on Reddit, with millions of active members who post constantly about games, players, trades, and league news. The discourse is highly varied: the same subreddit produces detailed statistical breakdowns, wild GOAT-debate hot takes, and pure in-the-moment reactions — sometimes all in the same game thread. This range makes it an ideal fit for a classification task, because the differences between post types are real and meaningful to the community itself. NBA fans regularly call out bad takes, praise good analysis, and dismiss pure emotional reactions — they already have informal mental categories that this classifier formalizes.

**Why it works for classification:**
- Posts vary enormously in structure and substance
- The distinction between "evidence-backed argument" and "confident opinion" is one the community itself cares about
- Reaction posts are clearly distinct in tone and length from the other two types
- Data is abundant and fully public

---

## 2. Label Taxonomy

### Label 1: `analysis`
**Definition:** A post that makes a structured argument supported by specific, verifiable evidence — statistics, historical comparisons, tactical observations, or play-by-play breakdowns. The reasoning would hold up even if you removed the opinion framing.

**Example 1:**
> "Jokic ranks in the 98th percentile for passes out of post-ups this season, leading to the second-highest teammate shot quality in the league. His MVP case isn't just narrative — the numbers show his impact extends well beyond his own scoring."

**Example 2:**
> "Every NBA champion since 2015 has finished in the top 5 in defensive rating. The Knicks are currently ranked 17th. History says that's a problem that doesn't just fix itself in the playoffs."

---

### Label 2: `hot_take`
**Definition:** A bold, confident opinion stated without meaningful supporting evidence. The post makes a strong claim and asserts it directly rather than arguing for it. The claim may be true or false, but the post doesn't try to prove it.

**Example 1:**
> "LeBron has never been the best player on any of his championship teams. Jordan never needed a superteam. The GOAT debate isn't even close."

**Example 2:**
> "Steph Curry is the most overrated player of his generation. The Warriors system made him. Put him on the Pistons and nobody is talking about him."

---

### Label 3: `reaction`
**Definition:** An immediate emotional response to a specific moment, game, or piece of news. The post expresses a feeling rather than making any kind of argument. Typically short, often uses caps or exclamation points, and is clearly tied to something that just happened.

**Example 1:**
> "LETS GOOOOOOO HE ACTUALLY DID IT I AM SCREAMING"

**Example 2:**
> "I can't believe what I just watched. This team is going to give me a heart attack. What a game."

---

## 3. Hard Edge Cases

### The primary edge case: The One-Stat Hot Take

**Example post:**
> "LeBron's playoff win rate against top-seeded opponents is below .500. He's overrated."

**Why it's ambiguous:** This post mentions a real, specific statistic — which looks like `analysis`. But the stat is cherry-picked to support an insult rather than being part of a genuine argument. There is no context, no comparison, and no reasoning beyond the number itself.

**Decision rule:** If the evidence would hold up as an argument on its own after removing the opinion framing → `analysis`. If the stat exists only to decorate an assertion, and removing the opinion would leave nothing worth reading → `hot_take`. The post above is `hot_take` because the stat is a prop, not a proof.

---

### Secondary edge case: The Analytical Reaction

**Example post:**
> "68% true shooting as a team?? That's the most efficient offensive game I've seen all season. Unbelievable."

**Why it's ambiguous:** This post mentions a specific stat, which looks like `analysis`. But the post is responding to a specific moment with amazement — the number is dropped in as an expression of disbelief, not as part of a reasoned argument.

**Decision rule:** If the post is responding emotionally to a specific live event and the statistic amplifies the feeling rather than supporting a claim → `reaction`. If the post uses the stat to build toward a conclusion → `analysis`. The post above is `reaction`.

---

### Tertiary edge case: The Thin Analysis

**Example post:**
> "Wembanyama leads all rookies in blocks and steals per game. He's going to be special."

**Why it's ambiguous:** It has a stat and makes a claim, but the connection between the two is too thin to qualify as a real argument. One undeveloped stat does not an analysis make.

**Decision rule:** Ask — would a reader who disagreed find this convincing? If the evidence is sufficient to support the conclusion on its own, it's `analysis`. If you'd need more to be persuaded, it's `hot_take`. The post above is `hot_take` (the stat alone doesn't prove the conclusion, and the conclusion is vague).

---

## 4. Data Collection Plan

### Source
All examples will be collected from public posts and comments on r/nba. No authentication required. Three content types will be targeted for each label:

| Label | Primary Source |
|---|---|
| `reaction` | Game Threads and Post-Game Threads (top-level comments) |
| `hot_take` | Posts with "unpopular opinion," "change my mind," "GOAT debate," or "overrated" in the title; top-level comments in heavily upvoted opinion posts |
| `analysis` | Posts tagged with "Analysis" flair; posts containing stats or tactical breakdowns; weekly discussion threads |

### Collection targets

| Label | Target count |
|---|---|
| `analysis` | 70–75 |
| `hot_take` | 70–75 |
| `reaction` | 70–75 |
| **Total** | **210–225** |

Collecting slightly over 200 gives buffer for any examples that get dropped during validation (duplicates, too short, unclear).

### If a label is underrepresented
If any label falls below 60 examples after the initial collection pass, I will do a targeted second pass for that label specifically before beginning annotation. I will not proceed to fine-tuning with any label below 60 examples, as this risks training a model that ignores the underrepresented class.

### Data handling
- Collected manually (copy-paste into Google Sheets, exported as CSV)
- No private content, no authenticated sources
- Final file: `dataset.csv` with columns `text`, `label`, `notes`
- Single file — train/val/test split handled by the notebook (70/15/15)

---

## 5. Evaluation Metrics

### Metrics I will use

**Primary metric: per-class F1 score**
F1 is the harmonic mean of precision and recall. It is the right metric here because:
- The dataset may not be perfectly balanced, and accuracy rewards predicting the majority class
- F1 penalizes both over-predicting a label (low precision) and under-predicting it (low recall)
- If the model completely fails on one label (F1 ≈ 0), that should be visible in the report even if overall accuracy looks acceptable

**Secondary metric: overall accuracy**
Accuracy gives a quick top-line comparison between the baseline and the fine-tuned model and is easy to communicate.

**Supporting tool: confusion matrix**
The confusion matrix shows which specific label pairs are being confused and in which direction. A directional error pattern (e.g., the model consistently calls `hot_take` → `analysis`) tells me exactly which boundary the model hasn't learned and gives me something concrete to analyze in the evaluation report.

### Why accuracy alone is not enough
A model that predicts `reaction` for every post could achieve 33% accuracy on a balanced 3-class dataset — matching a random guesser. If the classes are imbalanced and one label appears in 50% of posts, a model predicting only that label gets 50% accuracy without learning anything. Per-class F1 catches this; accuracy does not.

---

## 6. Definition of Success

A classifier is **genuinely useful** if it could be deployed as a lightweight moderation or content-tagging tool that helps community moderators or users sort posts by type. For that to be true:

| Criterion | Threshold |
|---|---|
| Fine-tuned model accuracy | At least 15 percentage points above baseline |
| Per-class F1 (all classes) | No class below 0.60 |
| Worst-case class F1 | At least 0.55 (model is learning something about every class) |
| Baseline comparison | Fine-tuned model strictly outperforms zero-shot Groq |

If the fine-tuned model fails to beat the baseline by at least 15 points, or if any single class has F1 below 0.55, I will treat the classifier as not deployment-ready and explain why in the evaluation report.

These thresholds are specific enough to evaluate objectively at the end of the project.

---

## 7. AI Tool Plan

### a) Label stress-testing
**What I'll do:** Give Claude my three label definitions and edge case descriptions, and ask it to generate 10–15 posts that sit at the boundary between two labels — particularly `analysis` vs. `hot_take`.

**Goal:** If Claude produces posts I can't classify cleanly using my decision rules, that is a signal that the definitions need tightening before I annotate 200 examples.

**When:** Before starting annotation. I will not begin labeling until I can classify all stress-test posts without ambiguity.

### b) Annotation assistance
**What I'll do:** After collecting all 200+ raw posts, I may use an LLM to pre-label a batch of examples by providing my label definitions and asking for one label per post.

**Safeguard:** I will review and correct every pre-assigned label individually. Any example I am not confident about after review will be re-read from scratch without the pre-label visible. Pre-labeled examples will be tracked in the `notes` column with a flag like `[pre-labeled]`.

**Disclosure:** If I use this workflow, I will disclose it in the AI Usage section of the README.

### c) Failure analysis
**What I'll do:** After fine-tuning, paste my full list of wrong predictions into Claude and ask it to identify any common themes — similar post length, label pairs that are consistently confused, posts with sarcasm, short or ambiguous posts, etc.

**Goal:** Surface patterns I might miss reviewing examples one at a time.

**Verification:** I will re-read every example in any pattern Claude identifies and confirm the pattern myself before writing it into the evaluation report. I will note what Claude suggested and what I verified or corrected.

---

## 8. Hard Annotation Decisions (updated during labeling)

*This section will be filled in during Milestone 3 as difficult examples are encountered. The spec requires at least 3 documented cases.*

### Difficult example 1
> **Post:** *(to be filled in during annotation)*
> **Candidate labels:** *(e.g., analysis vs. hot_take)*
> **Decision:** *(label chosen)*
> **Reasoning:** *(why)*

### Difficult example 2
> **Post:** *(to be filled in during annotation)*
> **Candidate labels:**
> **Decision:**
> **Reasoning:**

### Difficult example 3
> **Post:** *(to be filled in during annotation)*
> **Candidate labels:**
> **Decision:**
> **Reasoning:**

---

## 9. Stretch Features (update before starting each one)

*Check any stretch features planned and update this section before beginning work on them.*

- [ ] Inter-annotator reliability
- [ ] Confidence calibration
- [ ] Error pattern analysis
- [ ] Deployed interface

---

*Last updated: before data collection (Milestone 2)*