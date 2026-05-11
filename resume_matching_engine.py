import math

# =============================================================================
# SKILL ALIASES — DO NOT MODIFY
# =============================================================================
SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# =============================================================================
# DATASET
# =============================================================================
RESUMES = {
    "Arjun Sharma":    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair":      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta":     "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel":     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh":    "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta":     "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao":     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar":    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer":      "python, R, statistics, ML, regression, clustering, Power-BI",
}

JDS = {
    "JD-1 — Kakao (ML Engineer)": [
        "python", "machine_learning", "deep_learning", "tensorflow", "pytorch",
        "sql", "data_visualization", "nlp", "bert", "feature_engineering", "statistics"
    ],
    "JD-2 — Naver (Backend Engineer)": [
        "java", "spring_boot", "mysql", "postgresql", "microservices",
        "docker", "kubernetes", "rest_api", "ci_cd", "redis"
    ],
    "JD-3 — Line (Frontend Engineer)": [
        "javascript", "react", "vue", "typescript", "rest_api",
        "html_css", "nodejs", "graphql", "redux", "jest", "aws"
    ],
}

# =============================================================================
# STEP 1 — NORMALIZE SKILLS
# Sort alias keys by length descending so multi-word phrases match before
# single tokens (e.g. "spring boot" matches before "spring" or "boot")
# =============================================================================
SORTED_ALIAS_KEYS = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)

def normalize_skills(raw):
    tokens = [t.strip().lower() for t in raw.split(",")]
    result = []
    seen = set()
    for token in tokens:
        for alias_key in SORTED_ALIAS_KEYS:
            if token == alias_key:
                canonical = SKILL_ALIASES[alias_key]
                # Deduplication: only add if not seen before
                if canonical not in seen:
                    result.append(canonical)
                    seen.add(canonical)
                break
        # No match -> discard token (as per problem spec)
    return result

# =============================================================================
# STEP 2 — BUILD VOCABULARY
# Union of all normalized skills across all resumes, sorted alphabetically
# =============================================================================
def build_vocabulary(normalized_resumes):
    vocab_set = set()
    for skills in normalized_resumes.values():
        vocab_set.update(skills)
    return sorted(vocab_set)

# =============================================================================
# STEP 3 — COMPUTE TF-IDF
# TF = 1/N (after dedup, every skill appears exactly once)
# IDF = ln(10 / df)  — natural log, no smoothing
# TF-IDF = TF * IDF
# =============================================================================
def compute_df(normalized_resumes, vocab):
    df = {}
    for skill in vocab:
        df[skill] = sum(1 for skills in normalized_resumes.values() if skill in skills)
    return df

def compute_tfidf_vector(skills, vocab, df):
    N = len(skills)
    vec = {}
    for skill in vocab:
        if skill in skills:
            tf = 1 / N
            idf = math.log(10 / df[skill])  # natural log
            vec[skill] = tf * idf
        else:
            vec[skill] = 0.0
    return vec

# =============================================================================
# STEP 4 — BUILD JD BINARY VECTORS
# 1 if skill in JD required+preferred, 0 otherwise
# =============================================================================
def build_jd_vector(jd_skills, vocab):
    return {skill: (1 if skill in jd_skills else 0) for skill in vocab}

# =============================================================================
# STEP 5 — COSINE SIMILARITY
# Cosine(A, B) = dot(A,B) / (|A| * |B|)
# A = resume TF-IDF vector, B = JD binary vector
# =============================================================================
def cosine_similarity(tfidf_vec, jd_vec, vocab):
    dot = sum(tfidf_vec[s] * jd_vec[s] for s in vocab)
    norm_a = math.sqrt(sum(tfidf_vec[s] ** 2 for s in vocab))
    norm_b = math.sqrt(sum(jd_vec[s] ** 2 for s in vocab))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# =============================================================================
# MAIN
# =============================================================================
def main():
    # --- Normalize ---
    print("=" * 60)
    print("STEP 1 — SKILL NORMALIZATION")
    print("=" * 60)
    normalized = {}
    for name, raw in RESUMES.items():
        normalized[name] = normalize_skills(raw)
        print(f"{name}: {normalized[name]}")
        print(f"  N (unique skills) = {len(normalized[name])}")

    # --- Vocabulary ---
    print("\n" + "=" * 60)
    print("STEP 2 — VOCABULARY")
    print("=" * 60)
    vocab = build_vocabulary(normalized)
    print(f"Total terms: {len(vocab)}")
    for i, term in enumerate(vocab):
        print(f"  {i:2d}. {term}")

    # --- IDF ---
    print("\n" + "=" * 60)
    print("STEP 3 — IDF VALUES")
    print("=" * 60)
    df = compute_df(normalized, vocab)
    for skill in vocab:
        idf_val = math.log(10 / df[skill])
        print(f"  {skill}: df={df[skill]}, IDF={idf_val:.6f}")

    # --- TF-IDF vectors ---
    tfidf_vectors = {}
    for name, skills in normalized.items():
        tfidf_vectors[name] = compute_tfidf_vector(skills, vocab, df)

    # --- JD vectors ---
    print("\n" + "=" * 60)
    print("STEP 4 — JD BINARY VECTORS")
    print("=" * 60)
    jd_vectors = {}
    for jd_name, jd_skills in JDS.items():
        jd_vectors[jd_name] = build_jd_vector(jd_skills, vocab)
        matched = [s for s in jd_skills if s in vocab]
        unmatched = [s for s in jd_skills if s not in vocab]
        print(f"{jd_name}")
        print(f"  In vocab : {matched}")
        print(f"  Not found: {unmatched}")

    # --- Rankings ---
    print("\n" + "=" * 60)
    print("STEP 5 — FULL RANKINGS")
    print("=" * 60)
    all_scores = {}
    for jd_name, jd_vec in jd_vectors.items():
        scores = []
        for name in RESUMES:
            score = cosine_similarity(tfidf_vectors[name], jd_vec, vocab)
            scores.append((name, score))
        scores.sort(key=lambda x: (-x[1], x[0]))  # desc score, alpha tiebreak
        all_scores[jd_name] = scores
        print(f"\n{jd_name}")
        for rank, (name, score) in enumerate(scores, 1):
            print(f"  {rank:2d}. {name}: {score:.6f}")

    # --- Final output ---
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    for jd_name, scores in all_scores.items():
        top3 = scores[:3]
        print(f"\n{jd_name}")
        print(", ".join(f"{name}({score:.2f})" for name, score in top3))

if __name__ == "__main__":
    main()
