# Resume Matching Engine
### Redrob AI Campus Hackathon — Individual Competition
**Powered by McKinley Rice**

---

## Problem Statement

Given 10 resume datasets from Indian university students and 3 Job Descriptions (JDs) from Korean technology companies, build a program that:

- Normalizes noisy resume skill data
- Computes TF-IDF vectors for resumes
- Builds binary vectors for job descriptions
- Calculates cosine similarity between resumes and JDs
- Outputs the Top 3 matching candidates per JD

---

## Tech Stack

| Item | Detail |
|---|---|
| Language | Python 3 |
| Libraries | `math`, `re` (standard library only) |
| External Libraries | None (prohibited by problem spec) |
| AI Tool Used | Redrob AI |

---

## Project Structure

```
resume_matching_engine.py   # Complete solution — all steps in one script
README.md                   # This file
```

---

## How It Works — Pipeline

### Step 1 — Skill Normalization

Each resume's raw skill string is processed as follows:

1. Split on commas
2. Strip whitespace and lowercase every token
3. Sort `SKILL_ALIASES` keys by length **descending** — this ensures multi-word phrases (`"spring boot"`, `"feature engineering"`, `"data structure"`, `"competitive programming"`) are matched **before** single tokens
4. Match each token against the alias map; first match wins
5. Discard any token not found in the alias map
6. Deduplicate — each canonical skill appears only once per resume

**Critical edge case:** `"data-viz"` and `"matplotlib"` both map to `data_visualization`. After deduplication, Sneha Patel has **N=6**, not 7. This directly affects her TF score.

---

### Step 2 — Vocabulary Construction

- Union of all normalized, deduplicated skills across all 10 resumes
- Sorted **alphabetically**
- Same ordering used for both resume TF-IDF vectors and JD binary vectors
- Total: **48 unique terms**

---

### Step 3 — TF-IDF Computation (Resumes Only)

Exact formulas as specified:

```
TF(skill, resume) = 1 / N
    where N = total unique skills in that resume after deduplication

IDF(skill) = ln(10 / df(skill))
    where df(skill) = number of resumes containing the skill
    natural log (math.log), no smoothing

TF-IDF = TF × IDF
```

Notable IDF values:
- `python` — df=6, IDF=0.5108 (common, low weight)
- `data_visualization` — df=3, IDF=1.2040
- Most skills — df=1, IDF=2.3026 (rare, high discriminating power)
- `pytorch`, `redis` — not present in any resume (IDF undefined, excluded from vocab)

---

### Step 4 — JD Binary Vectors

JD skills (required + preferred) are normalized through the same `SKILL_ALIASES` map and matched against the vocabulary.

| JD | Terms in Vocab | Terms Not Found |
|---|---|---|
| JD-1 Kakao ML Engineer | 10/11 | `pytorch` |
| JD-2 Naver Backend Engineer | 9/10 | `redis` |
| JD-3 Line Frontend Engineer | 11/11 | — |

Binary vector: `1` if skill present, `0` otherwise.

---

### Step 5 — Cosine Similarity & Ranking

```
Cosine(A, B) = dot(A, B) / (|A| × |B|)

    A = Resume TF-IDF vector
    B = JD binary vector
    |A| = Euclidean norm of A
    |B| = Euclidean norm of B
```

Tie-breaking: alphabetical by candidate name (ascending).

---

## Results

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## Full Rankings

### JD-1 — Kakao (ML Engineer)
| Rank | Candidate | Score |
|---|---|---|
| 1 | Sneha Patel | 0.569570 |
| 2 | Karan Mehta | 0.534143 |
| 3 | Arjun Sharma | 0.395838 |
| 4 | Meera Iyer | 0.334485 |
| 5 | Vikram Singh | 0.034864 |
| 6 | Ananya Krishnan | 0.029812 |
| 7 | Aditya Kumar | 0.000000 |
| 8 | Deepika Rao | 0.000000 |
| 9 | Priya Nair | 0.000000 |
| 10 | Rahul Gupta | 0.000000 |

### JD-2 — Naver (Backend Engineer)
| Rank | Candidate | Score |
|---|---|---|
| 1 | Rahul Gupta | 0.810860 |
| 2 | Ananya Krishnan | 0.283296 |
| 3 | Deepika Rao | 0.190599 |
| 4 | Priya Nair | 0.117167 |
| 5 | Aditya Kumar | 0.000000 |
| 6 | Arjun Sharma | 0.000000 |
| 7 | Karan Mehta | 0.000000 |
| 8 | Meera Iyer | 0.000000 |
| 9 | Sneha Patel | 0.000000 |
| 10 | Vikram Singh | 0.000000 |

### JD-3 — Line (Frontend Engineer)
| Rank | Candidate | Score |
|---|---|---|
| 1 | Aditya Kumar | 0.665711 |
| 2 | Priya Nair | 0.575552 |
| 3 | Ananya Krishnan | 0.345807 |
| 4 | Deepika Rao | 0.086202 |
| 5 | Arjun Sharma | 0.000000 |
| 6 | Karan Mehta | 0.000000 |
| 7 | Meera Iyer | 0.000000 |
| 8 | Rahul Gupta | 0.000000 |
| 9 | Sneha Patel | 0.000000 |
| 10 | Vikram Singh | 0.000000 |

---

## Key Design Decisions

**Multi-word phrase priority** — Alias keys sorted by length descending before matching. Without this, `"spring boot"` gets tokenized into `"spring"` (no match) and `"boot"` (no match), silently discarding Rahul Gupta's most relevant skill for JD-2.

**Deduplication before TF** — Since TF = 1/N after dedup, N must reflect unique canonical skills only. Sneha Patel's raw input has both `data-viz` and `matplotlib`, both mapping to `data_visualization`. Counting both inflates N and deflates TF incorrectly.

**No smoothing on IDF** — Problem spec explicitly prohibits it. `IDF = ln(10 / df)` exactly, using Python's `math.log()` which defaults to natural log.

**Vocabulary from resumes only** — JD skills are NOT used to build the vocabulary. Skills like `pytorch` and `redis` appear only in JDs, so they contribute nothing to similarity computation. This is correct per spec.

---

## How to Run

```bash
python3 resume_matching_engine.py
```

No installation required. Pure Python standard library.

---

## Redrob AI Prompt Strategy

The solution was built using 3 staged prompts in Redrob AI:

**Prompt 1 — Normalization**
Normalize resume skills by splitting on commas, lowercasing, sorting SKILL_ALIASES keys by length descending for multi-word phrase priority, mapping through alias map, discarding unmatched tokens, and deduplicating per resume.

**Prompt 2 — Vocabulary + TF-IDF**
Build shared vocabulary from all normalized skills sorted alphabetically. Compute TF-IDF using TF=1/N, IDF=ln(10/df) with natural log and no smoothing. Print IDF table and per-candidate non-zero TF-IDF values.

**Prompt 3 — JD Vectors + Similarity + Rankings**
Build binary JD vectors over the same vocabulary. Compute cosine similarity for all 10 candidates × 3 JDs. Rank descending, break ties alphabetically, output top 3 per JD rounded to 2 decimal places.
