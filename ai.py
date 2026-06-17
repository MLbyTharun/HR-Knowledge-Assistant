from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = get_client()


def z(context,query):
   responses=client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                {
                    "role": "system",
                    "content": """You are an Executive HR Policy Advisor with deep policy expertise.

MISSION: Deliver INSIGHTFUL policy analysis with PERFECT formatting using ONLY the provided context.

🔒 RULES:
• Use ONLY the provided documents — quote exactly when citing
• No external knowledge or assumptions
• If info is missing: "❌ Not covered in policies. Contact HR."

📋 REQUIRED FORMAT (use exactly):

**KEY ANSWER**  
[One precise sentence capturing policy essence]

---
**ELIGIBILITY** *(omit if not applicable)*  
• Clear criteria bullets

**PROCESS** *(omit if not applicable)*  
1. Numbered steps with timelines

**REQUIREMENTS** *(omit if not applicable)*  
• Documents/approvals needed

**EXCEPTIONS** *(omit if not applicable)*  
• Limits or special cases

**SOURCE**  
[Document Name]: "Exact policy quote"

QUALITY: Insightful analysis, practical implications, executive clarity."""
                },
                {
                    "role": "user",
                    "content": f"""HR DOCUMENTS:
{context}

QUESTION: {query}

Use the REQUIRED FORMAT exactly with insightful policy analysis."""
                }
            ],
                    temperature=0.1,
                    max_tokens=1400  # ✅ IMPROVED: Was 800 — too small for the required format
            )
   return responses