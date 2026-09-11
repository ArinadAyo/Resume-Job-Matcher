
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =========================
# LOAD MODELS AND DATA
# =========================

@st.cache_resource
def load_jobbert():
    return SentenceTransformer("TechWolf/JobBERT-v3")


@st.cache_data
def load_esco():
    return pd.read_csv(
        "skills_en.csv",
        sep=",",
        low_memory=False
    )


jobbert_model = load_jobbert()
skills_df = load_esco()


# =========================
# SKILL ALIASES
# =========================

skill_aliases = {
    "python": ["python"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "sql": ["sql"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "data analysis": ["data analysis", "data analytics"],
    "fastapi": ["fastapi", "fast api"],
    "docker": ["docker"]
}


technology_skills = [
    "Docker",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "PyTorch",
    "TensorFlow",
    "React",
    "Node.js",
    "Kubernetes",
    "Git"
]


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# SKILL EXTRACTION
# =========================

def extract_skills_with_regex(text, skill_aliases):

    found_skills = []

    for skill, aliases in skill_aliases.items():

        for alias in aliases:

            pattern = r"\b" + re.escape(alias) + r"\b"

            if re.search(pattern, text):
                found_skills.append(skill)
                break

    return found_skills


# =========================
# ESCO SKILL MATCHING
# =========================

def find_esco_skill(skill):

    skill = skill.strip().lower()

    matches = skills_df[
        skills_df["preferredLabel"]
        .fillna("")
        .str.strip()
        .str.lower() == skill
    ]

    if len(matches) > 0:

        match = matches.iloc[0]

        return {
            "input": skill,
            "matched_skill": match["preferredLabel"],
            "uri": match["conceptUri"],
            "match_type": "preferred_label"
        }

    for _, row in skills_df.iterrows():

        if pd.isna(row["altLabels"]):
            continue

        aliases = [
            alias.strip().lower()
            for alias in str(row["altLabels"]).split("\n")
        ]

        if skill in aliases:

            return {
                "input": skill,
                "matched_skill": row["preferredLabel"],
                "uri": row["conceptUri"],
                "match_type": "alternative_label"
            }

    return {
        "input": skill,
        "matched_skill": None,
        "uri": None,
        "match_type": "not_found"
    }


def resolve_skill(skill):

    esco_result = find_esco_skill(skill)

    if esco_result["matched_skill"] is not None:
        return esco_result

    for tech in technology_skills:

        if skill.lower() == tech.lower():

            return {
                "input": skill,
                "matched_skill": tech,
                "uri": None,
                "match_type": "technology"
            }

    return {
        "input": skill,
        "matched_skill": None,
        "uri": None,
        "match_type": "not_found"
    }


# =========================
# SCORING
# =========================

def calculate_match_score(tfidf, semantic, skill_coverage):

    return (
        tfidf * 0.20
        + semantic * 0.30
        + skill_coverage * 0.50
    )


def interpret_score(score):

    if score >= 80:
        return "Strong Match"

    elif score >= 60:
        return "Good Match"

    elif score >= 40:
        return "Moderate Match"

    else:
        return "Weak Match"


# =========================
# MAIN MATCHING FUNCTION
# =========================

def match_resume_to_job(resume_text, job_text):

    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_text)

    # Extract skills
    resume_skills = extract_skills_with_regex(
        clean_resume,
        skill_aliases
    )

    job_skills = extract_skills_with_regex(
        clean_job,
        skill_aliases
    )

    # Resolve through ESCO / technology vocabulary
    resume_resolved = []
    unknown_resume = []

    for skill in resume_skills:

        result = resolve_skill(skill)

        if result["matched_skill"] is not None:
            resume_resolved.append(
                result["matched_skill"].lower()
            )
        else:
            unknown_resume.append(skill)

    job_resolved = []
    unknown_job = []

    for skill in job_skills:

        result = resolve_skill(skill)

        if result["matched_skill"] is not None:
            job_resolved.append(
                result["matched_skill"].lower()
            )
        else:
            unknown_job.append(skill)

    # Remove duplicates
    resume_resolved = list(set(resume_resolved))
    job_resolved = list(set(job_resolved))

    # Matching and missing skills
    matching_skills = sorted(
        list(set(resume_resolved) & set(job_resolved))
    )

    missing_skills = sorted(
        list(set(job_resolved) - set(resume_resolved))
    )

    # Skill coverage
    if len(job_resolved) > 0:
        skill_coverage = (
            len(matching_skills) / len(job_resolved)
        )
    else:
        skill_coverage = 0

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_vectors = tfidf_vectorizer.fit_transform([
        clean_resume,
        clean_job
    ])

    tfidf_similarity = cosine_similarity(
        tfidf_vectors
    )[0][1]

    # JobBERT semantic similarity
    embeddings = jobbert_model.encode([
        clean_resume,
        clean_job
    ])

    semantic_similarity = cosine_similarity(
        embeddings
    )[0][1]

    # Final score
    final_score = calculate_match_score(
        tfidf_similarity,
        semantic_similarity,
        skill_coverage
    )

    assessment = interpret_score(
        final_score * 100
    )

    return {
        "final_score": final_score * 100,
        "assessment": assessment,
        "tfidf_score": tfidf_similarity * 100,
        "semantic_score": semantic_similarity * 100,
        "skill_coverage": skill_coverage * 100,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "unknown_resume_skills": unknown_resume,
        "unknown_job_skills": unknown_job
    }


# =========================
# STREAMLIT INTERFACE
# =========================

st.set_page_config(
    page_title="AI Resume Job Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume ↔ Job Matcher")

st.write(
    "Compare a resume with a job description using "
    "semantic similarity and skill matching."
)

resume_text = st.text_area(
    "Resume",
    height=300,
    placeholder="Paste your resume here..."
)

job_text = st.text_area(
    "Job Description",
    height=300,
    placeholder="Paste the job description here..."
)

if st.button("Analyze Match"):

    if not resume_text.strip() or not job_text.strip():

        st.error("Please provide both a resume and a job description.")

    else:

        with st.spinner("Analyzing..."):

            result = match_resume_to_job(
                resume_text,
                job_text
            )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Match Score",
            f"{result['final_score']:.2f}%"
        )

        col2.metric(
            "Skill Coverage",
            f"{result['skill_coverage']:.2f}%"
        )

        col3.metric(
            "Assessment",
            result["assessment"]
        )

        st.subheader("Matching Skills")

        if result["matching_skills"]:
            st.write(", ".join(result["matching_skills"]))
        else:
            st.write("None identified.")

        st.subheader("Missing Skills")

        if result["missing_skills"]:
            st.write(", ".join(result["missing_skills"]))
        else:
            st.write("None identified.")

        st.subheader("Additional Details")

        st.write(
            f"TF-IDF similarity: {result['tfidf_score']:.2f}%"
        )

        st.write(
            f"JobBERT semantic similarity: "
            f"{result['semantic_score']:.2f}%"
        )

        if result["unknown_resume_skills"]:
            st.write(
                "Unknown resume skills:",
                ", ".join(result["unknown_resume_skills"])
            )

        if result["unknown_job_skills"]:
            st.write(
                "Unknown job skills:",
                ", ".join(result["unknown_job_skills"])
            )
