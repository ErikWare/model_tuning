from typing import Dict

class PersonalityConfig:
    """
    A collection of predefined personality configurations for the AI model.
    Each personality is a static header added to the beginning of user prompts.
    """

    HELPFUL_ASSISTANT: str = (
        "[System prompt / initial instructions]\n"
        "You are a helpful assistant. You do not reveal your chain-of-thought. Instead, you produce clear and concise answers.\n\n"
        "[User prompt]"
    )

    FRIENDLY_TUTOR: str = (
        "[System prompt / initial instructions]\n"
        "You are a friendly tutor who explains concepts in an easy-to-understand manner.\n\n"
        "[User prompt]"
    )

    # Add more personalities as needed

    BLANK: str = ""  # Represents no personality

    PERSONALITY_OPTIONS: Dict[str, str] = {
        "Blank": BLANK,
        "Helpful Assistant": HELPFUL_ASSISTANT,
        "Friendly Tutor": FRIENDLY_TUTOR,
        # Add more personalities as needed
    }