
import re
import io
from datetime import datetime

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Resume Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F5F7FA;
    color: #172B45;
}

[data-testid="stSidebar"] {
    background: #172B45;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 5px 0;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172B45;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 15px;
    color: #718096;
    margin-bottom: 28px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #172B45;
    margin: 18px 0 12px 0;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E4E9F0;
    border-radius: 12px;
    padding: 22px;
    min-height: 125px;
}

.metric-label {
    font-size: 13px;
    color: #718096;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #172B45;
    margin-top: 8px;
}

.panel {
    background: #FFFFFF;
    border: 1px solid #E4E9F0;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 18px;
}

.small-note {
    color: #718096;
    font-size: 13px;
}

.stButton > button {
    background: #3567A5;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 11px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #254F85;
    color: white;
    border: none;
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #AAB8C8;
    border-radius: 10px;
    padding: 15px;
}

hr {
    border-color: #E4E9F0;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "navigation" not in st.session_state:
    st.session_state.navigation = "Dashboard"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if st.session_state.pop("navigate_to_analysis", False):
    st.session_state.navigation = "Resume Analysis"


# --------------------------------------------------
# SKILL DATABASE
# --------------------------------------------------

SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript",
    "html", "css", "react", "node.js", "sql", "mysql",
    "mongodb", "machine learning", "deep learning",
    "artificial intelligence", "data analysis", "data science",
    "power bi", "tableau", "excel", "aws", "azure",
    "google cloud", "docker", "kubernetes", "git", "github",
    "flask", "django", "streamlit", "fastapi",
    "communication", "leadership", "problem solving",
    "teamwork", "project management", "linux",
    "rest api", "data structures", "algorithms",
    "cybersecurity", "networking", "figma"
]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def extract_resume_text(uploaded_file):
    """Extract text from an uploaded PDF."""

    try:
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)

        if reader.is_encrypted:
            return None, "This PDF is password protected."

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return None, "No readable text was found in this PDF."

        return text, None

    except Exception:
        return None, "Unable to read this PDF. Please upload a valid PDF file."


def calculate_similarity(resume_text, job_description):
    """Calculate text similarity using TF-IDF."""

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform([
        resume_text,
        job_description
    ])

    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return round(float(score) * 100, 1)


def extract_skills(text):
    """Find supported skills in text."""

    text_lower = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"

        if re.search(pattern, text_lower):
            found.append(skill.title())

    return sorted(set(found))


def detect_sections(text):
    """Detect common resume sections."""

    section_keywords = {
        "Contact Information": ["email", "phone", "mobile", "linkedin"],
        "Professional Summary": ["summary", "objective", "profile"],
        "Education": ["education", "qualification", "academic"],
        "Work Experience": ["experience", "employment", "work history"],
        "Projects": ["projects", "personal projects"],
        "Skills": ["skills", "technical skills"],
        "Certifications": ["certifications", "certificates", "courses"]
    }

    text_lower = text.lower()
    found_sections = []

    for section, keywords in section_keywords.items():
        if any(word in text_lower for word in keywords):
            found_sections.append(section)

    return found_sections


def generate_recommendations(
    resume_text,
    job_description,
    matched_skills,
    missing_skills,
    sections
):
    recommendations = []

    if len(resume_text.split()) < 150:
        recommendations.append(
            "Your resume appears short. Add relevant achievements, "
            "projects, and experience where appropriate."
        )

    if "Projects" not in sections:
        recommendations.append(
            "Add a Projects section with your academic or personal projects."
        )

    if "Education" not in sections:
        recommendations.append(
            "Include your educational qualifications and graduation details."
        )

    if "Skills" not in sections:
        recommendations.append(
            "Add a dedicated Skills section containing relevant technical skills."
        )

    if "Professional Summary" not in sections:
        recommendations.append(
            "Add a short professional summary tailored to the target role."
        )

    if missing_skills:
        recommendations.append(
            "Consider adding relevant missing skills only if you genuinely "
            "have those skills or experience."
        )

    if "Work Experience" not in sections:
        recommendations.append(
            "If you have internship, volunteering, or work experience, "
            "include it with measurable contributions."
        )

    if not recommendations:
        recommendations.append(
            "Your resume includes the main sections detected by this tool. "
            "Review each section for clarity and relevance."
        )

    return recommendations


def create_report(result):
    """Create a downloadable text report."""

    report = f"""
RESUME OPTIMIZER
Resume Analysis Report
Generated: {datetime.now().strftime("%d %B %Y, %I:%M %p")}

----------------------------------------
ANALYSIS SUMMARY
----------------------------------------

Target Role: {result['job_title']}
Text Similarity: {result['score']}%

Matched Skills:
{', '.join(result['matched']) if result['matched'] else 'None detected'}

Potentially Missing Skills:
{', '.join(result['missing']) if result['missing'] else 'None detected'}

Detected Resume Sections:
{', '.join(result['sections']) if result['sections'] else 'None detected'}

----------------------------------------
RECOMMENDATIONS
----------------------------------------

"""

    for number, recommendation in enumerate(result["recommendations"], 1):
        report += f"{number}. {recommendation}\n"

    report += """

----------------------------------------
IMPORTANT NOTE
----------------------------------------

The similarity percentage is a text comparison indicator.
It is not an official ATS score, hiring decision, or guarantee
of selection. Skill detection is based on a limited vocabulary.
Please verify all recommendations before editing your resume.
"""

    return report


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="padding: 15px 5px 25px 5px;">
        <div style="font-size: 28px; font-weight: 800;">
            Resume <span style="color:#75A9E8;">Optimizer</span>
        </div>
        <div style="font-size: 12px; color:#B9C8DA;">
            PROFESSIONAL RESUME ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    selected_page = st.radio(
        "Navigation",
        ["Dashboard", "Resume Analysis", "How It Works"],
        label_visibility="collapsed",
        key="navigation"
    )

    st.divider()

    st.markdown(
        "<div style='font-size:12px;color:#B9C8DA;'>"
        "RESUME OPTIMIZER v1.0<br><br>"
        "Built for smarter career preparation."
        "</div>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if selected_page == "Dashboard":

    st.markdown(
        '<div class="main-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">A clearer way to review and improve your resume.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h2 style="color:#172B45;">Make your resume more relevant.</h2>
        <p style="color:#718096;font-size:15px;">
        Compare your resume with a job description, review relevant skills,
        and identify areas that may need improvement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">DOCUMENT SUPPORT</div>
            <div class="metric-value">PDF</div>
            <div class="small-note">Resume upload format</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">SKILL REVIEW</div>
            <div class="metric-value">40+</div>
            <div class="small-note">Built-in skill keywords</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ANALYSIS</div>
            <div class="metric-value">3</div>
            <div class="small-note">Core review areas</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Get started</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload your resume and compare it with the requirements "
        "of the position you want to apply for."
    )

    if st.button("Open resume analysis", use_container_width=True):
        st.session_state.navigate_to_analysis = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Your resume is processed in memory by this app. "
        "Avoid uploading documents containing information you do not "
        "want to share with the hosting service."
    )


# --------------------------------------------------
# RESUME ANALYSIS
# --------------------------------------------------

elif selected_page == "Resume Analysis":

    st.markdown(
        '<div class="main-title">Resume Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Compare your resume with a target job description.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1])

    with left:

        st.markdown(
            '<div class="section-title">1. Upload your resume</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose your resume in PDF format",
            type=["pdf"],
            help="Maximum file size: 10 MB"
        )

        if uploaded_file and uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size exceeds 10 MB. Please upload a smaller PDF.")
            uploaded_file = None

    with right:

        st.markdown(
            '<div class="section-title">2. Target job details</div>',
            unsafe_allow_html=True
        )

        job_title = st.text_input(
            "Job title",
            placeholder="e.g. Python Developer"
        )

        job_description = st.text_area(
            "Job description",
            height=180,
            placeholder="Paste the complete job description here..."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_button = st.button(
        "Analyze my resume",
        use_container_width=True
    )

    if analyze_button:

        if not uploaded_file:
            st.error("Please upload your resume first.")

        elif not job_description.strip():
            st.error("Please enter the job description.")

        else:

            with st.spinner("Analyzing your resume..."):

                resume_text, error = extract_resume_text(uploaded_file)

                if error:
                    st.error(error)

                else:

                    try:
                        score = calculate_similarity(
                            resume_text,
                            job_description
                        )

                        resume_skills = extract_skills(resume_text)
                        job_skills = extract_skills(job_description)

                        matched = [
                            skill for skill in job_skills
                            if skill.lower() in
                            [item.lower() for item in resume_skills]
                        ]

                        missing = [
                            skill for skill in job_skills
                            if skill.lower() not in
                            [item.lower() for item in resume_skills]
                        ]

                        sections = detect_sections(resume_text)

                        recommendations = generate_recommendations(
                            resume_text,
                            job_description,
                            matched,
                            missing,
                            sections
                        )

                        st.session_state.analysis_result = {
                            "score": score,
                            "job_title": job_title.strip() or "Not specified",
                            "matched": matched,
                            "missing": missing,
                            "sections": sections,
                            "recommendations": recommendations
                        }

                    except ValueError:
                        st.error(
                            "The uploaded resume or job description "
                            "does not contain enough readable text to compare."
                        )

    # DISPLAY RESULTS

    result = st.session_state.analysis_result

    if result:

        st.divider()

        st.markdown(
            '<div class="section-title">Analysis results</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TEXT SIMILARITY</div>
                <div class="metric-value">{result['score']}%</div>
                <div class="small-note">Resume vs job description</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MATCHED SKILLS</div>
                <div class="metric-value">{len(result['matched'])}</div>
                <div class="small-note">Detected in both texts</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">POTENTIALLY MISSING</div>
                <div class="metric-value">{len(result['missing'])}</div>
                <div class="small-note">Found in job description</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(result["score"] / 100)

        st.caption(
            "Similarity measures text overlap. It is not an official ATS score."
        )

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown(
                '<div class="section-title">Matched skills</div>',
                unsafe_allow_html=True
            )

            if result["matched"]:
                for skill in result["matched"]:
                    st.success(f"✓ {skill}")
            else:
                st.write("No matching skills detected.")

        with col_b:

            st.markdown(
                '<div class="section-title">Potentially missing skills</div>',
                unsafe_allow_html=True
            )

            if result["missing"]:
                for skill in result["missing"]:
                    st.warning(f"• {skill}")
            else:
                st.success("No missing skills detected from the supported list.")

        st.markdown(
            '<div class="section-title">Resume sections</div>',
            unsafe_allow_html=True
        )

        if result["sections"]:
            section_cols = st.columns(3)

            for index, section in enumerate(result["sections"]):
                with section_cols[index % 3]:
                    st.success(f"✓ {section}")
        else:
            st.info("No common resume sections were detected.")

        st.markdown(
            '<div class="section-title">Improvement recommendations</div>',
            unsafe_allow_html=True
        )

        for number, recommendation in enumerate(
            result["recommendations"], 1
        ):
            st.markdown(f"**{number}.** {recommendation}")

        report = create_report(result)

        st.download_button(
            label="Download analysis report",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        if st.button("Clear analysis", use_container_width=True):
            st.session_state.analysis_result = None
            st.rerun()


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

elif selected_page == "How It Works":

    st.markdown(
        '<div class="main-title">How It Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand how Resume Optimizer reviews your document.'
        '</div>',
        unsafe_allow_html=True
    )

    steps = [
        (
            "01",
            "Upload your resume",
            "Upload a readable PDF resume. The application extracts "
            "the available text from the document."
        ),
        (
            "02",
            "Enter job requirements",
            "Paste the job description and optionally enter the "
            "position title you are targeting."
        ),
        (
            "03",
            "Review the comparison",
            "The application compares text, checks supported skill "
            "keywords, and identifies common resume sections."
        ),
        (
            "04",
            "Improve your resume",
            "Use the recommendations as a guide and download a "
            "text report for later reference."
        )
    ]

    for number, title, description in steps:

        st.markdown(f"""
        <div class="panel">
            <div style="font-size:13px;font-weight:700;color:#3567A5;">
                STEP {number}
            </div>
            <h3 style="color:#172B45;margin-top:8px;">{title}</h3>
            <p style="color:#718096;margin-bottom:0;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Important information</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Resume Optimizer provides a basic text and keyword comparison. "
        "It does not reproduce the scoring systems of commercial ATS "
        "platforms, verify qualifications, or predict hiring outcomes. "
        "Always review the results yourself."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#8A96A8;font-size:12px;padding:10px;">
        © 2026 Resume Optimizer · Professional Resume Analysis
    </div>
    """,
    unsafe_allow_html=True
)
