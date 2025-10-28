import os
import json
import re
import time
from tqdm import tqdm
import types

# ============================================================
# LLM client bootstrap (env-driven; no keys in code)
# - Set USE_API=true and LLM_BACKEND=azure|openai to use real APIs
# - Otherwise falls back to Dummy (offline)
# Env examples:
#   # Azure:
#   export USE_API=true
#   export LLM_BACKEND=azure
#   export AZURE_OPENAI_API_KEY=...
#   export AZURE_OPENAI_ENDPOINT=https://<your>.openai.azure.com/
#   export AZURE_OPENAI_API_VERSION=2024-05-01-preview
#   export AZURE_OPENAI_DEPLOYMENT=gpt-4o
#
#   # OpenAI:
#   export USE_API=true
#   export LLM_BACKEND=openai
#   export OPENAI_API_KEY=...
#   export OPENAI_MODEL=gpt-4o-mini
#   # optional proxy/base:
#   # export OPENAI_API_BASE=https://api.openai.com/v1
# ============================================================

USE_API = os.getenv("USE_API", "false").lower() == "true"
BACKEND = os.getenv("LLM_BACKEND", "azure").lower()  # azure | openai | dummy

# Dummy LLM (offline-compatible)
class _DummyChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)

class _DummyChatCompletions:
    def create(self, model=None, temperature: float = 0.0, max_tokens: int = 256, messages=None, **kwargs):
        # Deterministic pseudo-response based on prompt hash (for reproducible offline runs)
        prompt = '\n'.join([m.get('content', '') for m in (messages or [])])
        h = abs(hash(prompt)) % 1000
        content = f"[DUMMY OUTPUT {h}] " + (prompt[:100].replace('\n', ' '))
        return types.SimpleNamespace(choices=[_DummyChoice(content)])

class _DummyResponses:
    def create(self, model=None, input=None, **kwargs):
        text = str(input)[:120] if input is not None else ''
        return types.SimpleNamespace(output_text=f"[DUMMY OUTPUT] {text}")

class DummyLLM:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_DummyChatCompletions())
        self.responses = _DummyResponses()

def get_llm_client():
    if USE_API:
        if BACKEND == "azure":
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )
        elif BACKEND == "openai":
            from openai import OpenAI
            base = os.getenv("OPENAI_API_BASE") or None
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base)
    # default: dummy offline client
    return DummyLLM()

# Unified model/deployment name to keep your existing call sites unchanged
DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") if BACKEND == "azure"
    else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
)

client = get_llm_client()

# ============================================================
# Scenario hierarchy (as provided)
# ============================================================
scenario_hierarchy = {
    "Relationship Crisis Scenarios": {
        "Romantic Relationships": [
            "Breakup/Divorce",
            "Infidelity Discovery",
            "Long-distance Relationship Strain",
            "Marriage Proposal Rejection",
            "Wedding Cancellation",
            "Partner's Sudden Death",
            "Toxic Relationship Realization",
            "Dating Violence/Abuse",
            "Stalking by Ex-partner",
            "Custody Battle"
        ],
        "Family Relationships": [
            "Family Disownment",
            "Inheritance Disputes",
            "Parent's Divorce Impact",
            "Sibling Betrayal",
            "Family Business Conflict",
            "Caregiver Burnout",
            "Family Member's Addiction",
            "Estrangement from Children",
            "In-law Conflicts",
            "Family Member's Terminal Illness"
        ]
    },
    "Career Crisis Scenarios": {
        "Employment Issues": [
            "Sudden Job Loss",
            "Workplace Bullying",
            "Career Path Dead-end",
            "Age Discrimination",
            "Sexual Harassment",
            "Wrongful Termination",
            "Demotion",
            "Team Conflict",
            "Performance Crisis",
            "Skill Obsolescence"
        ],
        "Business Issues": [
            "Business Failure",
            "Partnership Betrayal",
            "Bankruptcy",
            "Employee Layoffs",
            "Corporate Scandal",
            "Workplace Accident",
            "Legal Compliance Crisis",
            "Market Competition Crisis",
            "Product Failure",
            "Reputation Damage"
        ]
    },
    "Financial Crisis Scenarios": {
        "Investment Issues": [
            "Investment Loss",
            "Cryptocurrency Crash",
            "Real Estate Loss",
            "Stock Market Crash",
            "Ponzi Scheme Victim",
            "Trading Account Blow-up",
            "Failed Business Investment",
            "Partnership Fund Loss",
            "Retirement Fund Loss",
            "Inheritance Loss"
        ],
        "Debt Issues": [
            "Bankruptcy Filing",
            "Mortgage Default",
            "Student Loan Crisis",
            "Medical Debt",
            "Credit Card Debt",
            "Loan Shark Threats",
            "Gambling Debts",
            "Business Loan Default",
            "Car Loan Default",
            "Identity Theft Impact"
        ]
    },
    "Social Crisis Scenarios": {
        "Professional Social Issues": [
            "Workplace Ostracism",
            "Professional Reputation Damage",
            "Conference Presentation Failure",
            "Team Project Failure",
            "Leadership Challenge",
            "Office Politics Crisis",
            "Professional Association Rejection",
            "Mentor Relationship Breakdown",
            "Client Relationship Crisis",
            "Industry Blacklisting"
        ],
        "Personal Social Issues": [
            "Social Media Crisis",
            "Friend Group Exclusion",
            "Community Rejection",
            "Club/Organization Expulsion",
            "Neighbor Conflict",
            "Social Event Disaster",
            "Religious Community Exclusion",
            "Sports Team Rejection",
            "Volunteer Organization Conflict",
            "Cultural Group Ostracism"
        ]
    },
    "Health Crisis Scenarios": {
        "Physical Health": [
            "Chronic Illness Diagnosis",
            "Sudden Disability",
            "Sports Career-Ending Injury",
            "Cosmetic Surgery Gone Wrong",
            "Weight Crisis",
            "Eating Disorder",
            "Sleep Disorder",
            "Chronic Pain",
            "Terminal Illness",
            "Addiction Recovery"
        ],
        "Mental Health": [
            "Anxiety Attack",
            "Depression Episode",
            "PTSD Trigger",
            "Panic Attack",
            "Identity Crisis",
            "Burnout",
            "Impostor Syndrome",
            "OCD Episode",
            "Phobia Trigger",
            "Self-harm Urges"
        ]
    },
    "Life Transition Scenarios": {
        "Location Changes": [
            "Immigration Crisis",
            "Forced Relocation",
            "Cultural Adaptation",
            "Housing Crisis",
            "Neighborhood Change",
            "College Transition",
            "Retirement Location",
            "Military Deployment",
            "Rural-Urban Transition",
            "International Assignment"
        ],
        "Identity Changes": [
            "Gender Transition",
            "Sexual Orientation Coming Out",
            "Religious Faith Crisis",
            "Political Identity Crisis",
            "Career Identity Loss",
            "Empty Nest Syndrome",
            "Retirement Identity Crisis",
            "Cultural Identity Crisis",
            "Disability Adaptation",
            "Age Milestone Crisis"
        ]
    },
    "Education/Academic Crisis": {
        "Academic Performance": [
            "Academic Failure",
            "Thesis/Dissertation Crisis",
            "Exam Failure",
            "Academic Misconduct",
            "Scholarship Loss",
            "Research Project Failure",
            "Academic Rejection",
            "Grade Crisis",
            "Study Abroad Crisis",
            "Academic Probation"
        ],
        "Educational Environment": [
            "School Bullying",
            "Teacher Conflict",
            "Peer Group Rejection",
            "Learning Disability Crisis",
            "Academic Competition",
            "School Transfer Crisis",
            "College Application Rejection",
            "Educational Funding Loss",
            "Major Change Crisis",
            "Academic Career Crisis"
        ]
    }
}

# ============================================================
# Prompt builder
# ============================================================
def create_prompt_enhanced_scenario(scenario: str, num_profiles: int = 4) -> str:
    """
    Build a prompt that requests logically consistent, diverse user profiles for a given scenario.
    """
    prompt = f"""
I am conducting sociological research and need to generate {num_profiles} unique user profiles based on the following scenario: "{scenario}". Each profile should align with the scenario and maintain logical consistency between variables.

Scenario: {scenario}

Each profile must include:
- Age (make it consistent with the scenario and other fields)
- Gender (diverse unless the scenario implies a skew)
- Marital Status (should correlate with the scenario where applicable)
- Profession (diverse, realistic)
- Economic Status (realistic given the scenario)
- Health Status (aligned with the scenario)
- Education Level (consistent with age/profession)
- Mental Health Status (correlated with scenario stressors)
- Past Self-Harm History (consistent with mental health)
- Emotional State (current state grounded in the scenario)

Generation requirements:
1) Profiles must be unique across the set.
2) Keep values concise and realistic.
3) Output MUST be a single JSON array of objects with exactly these fields plus "Scenario".

Example (format ONLY):
[
  {{
    "Scenario": "{scenario}",
    "Age": "25-34 years",
    "Gender": "Female",
    "Marital Status": "Divorced",
    "Profession": "Teacher",
    "Economic Status": "Moderate",
    "Health Status": "Good",
    "Education Level": "Bachelor's",
    "Mental Health Status": "Mild Depression",
    "Past Self-Harm History": "None",
    "Emotional State": "Overwhelmed by financial instability after divorce; determined to rebuild."
  }}
]
"""
    return prompt

# ============================================================
# Profile generation using the client above
# ============================================================
def generate_user_profiles_fixed_scenario(scenario: str, num_profiles: int = 4, max_retries: int = 5):
    """
    Generate synthetic user profiles for a fixed scenario.
    Returns a Python list (parsed from JSON) on success, or an empty list on failure.
    """
    prompt = create_prompt_enhanced_scenario(scenario, num_profiles)
    generated_text = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a data generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            generated_text = response.choices[0].message.content or ""
            break
        except Exception as e:
            print(f"Generation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return []

    # Try to extract a JSON array. First prefer fenced block ```json ... ```
    try:
        match = re.search(r"```json\s*(.*?)```", generated_text, re.DOTALL | re.IGNORECASE)
        text = match.group(1) if match else generated_text
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except Exception:
        return []

def generate_all_data_in_order(hierarchy: dict, per_issue_profiles: int = 5):
    """
    Iterate through the hierarchy in order and collect generated profiles for each issue.
    """
    all_user_profiles = []
    total_issues = sum(len(issues) for subcats in hierarchy.values() for issues in subcats.values())
    with tqdm(total=total_issues, desc="Generating Data", unit="issue") as pbar:
        for category, subcategories in hierarchy.items():
            for subcategory, issues in subcategories.items():
                for issue in issues:
                    profiles = generate_user_profiles_fixed_scenario(issue, num_profiles=per_issue_profiles)
                    if profiles:
                        all_user_profiles.extend(profiles)
                    pbar.update(1)
    return all_user_profiles

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Profiles per leaf issue (defaults to 5 to mirror your original workflow)
    PER_ISSUE_PROFILES = int(os.getenv("PER_ISSUE_PROFILES", "5"))
    # Output file path; override via OUTPUT_FILE env var
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "ordered_scenario_generated_data_profiles.json")

    profiles = generate_all_data_in_order(scenario_hierarchy, per_issue_profiles=PER_ISSUE_PROFILES)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE)) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    print(f"Data generation completed and saved to '{OUTPUT_FILE}'. Total profiles: {len(profiles)}")
