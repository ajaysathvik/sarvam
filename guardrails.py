BLOCKED_TOPICS = ["violence", "hate", "illegal", "abuse", "drug", "weapon", "bomb", "kill", "suicide"]  # customize

def check_input(text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Block before hitting LLM."""
    # Simple keyword check OR call another LLM classifier
    for topic in BLOCKED_TOPICS:
        if topic in text.lower():
            return False, f"Topic not allowed: {topic}"
    return True, ""

def check_output(text: str) -> tuple[bool, str]:
    """Returns (is_safe, cleaned_text). Filter LLM response."""
    # e.g. remove PII, harmful content
    return True, text
