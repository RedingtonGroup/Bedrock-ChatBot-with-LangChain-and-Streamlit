import re
from typing import Optional, Tuple

class ContentModerator:
    """
    A class to implement basic content moderation for text.
    In a real-world scenario, this would involve more sophisticated
    techniques like using a dedicated moderation API (e.g., AWS Comprehend,
    Azure Content Moderator, Google Cloud Natural Language API) or a fine-tuned
    machine learning model.
    """
    def __init__(self):
        # Define a list of keywords to flag. This is a very basic approach.
        # For production, consider NLP libraries or moderation APIs.
        self.banned_keywords = [
            "hate speech", "violence", "self-harm", "illegal", "exploit",
            "offensive", "inappropriate", "swear", "abusive", "threaten"
        ]
        # self.topic_restrictions = { # Removed as requested
        #     "financial advice": ["invest", "stock market", "financial planning", "loan", "mortgage"],
        #     "medical advice": ["diagnose", "cure", "medicine", "symptom", "treatment"],
        #     "legal advice": ["lawsuit", "court", "legal opinion", "contract", "rights"]
        # }

    def moderate_content(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks the given text for banned keywords.
        Returns (True, "reason") if flagged, otherwise (False, None).
        """
        text_lower = text.lower()
        for keyword in self.banned_keywords:
            if keyword in text_lower:
                return True, f"contains '{keyword}'"
        return False, None

    # Removed:
    # def check_topic_restriction(self, text: str, restricted_topic: Optional[str] = None) -> bool:
    #     """
    #     Checks if the text falls into a restricted topic.
    #     This is a simplistic check and would need a more robust NLP model for accuracy.
    #     """
    #     if not restricted_topic or restricted_topic not in self.topic_restrictions:
    #         return False
    #
    #     text_lower = text.lower()
    #     for keyword in self.topic_restrictions[restricted_topic]:
    #         if keyword in text_lower:
    #             return True
    #     return False

    # Removed:
    # def enforce_length_limit(self, text: str, max_length: int) -> str:
    #     """
    #     Truncates the text if it exceeds the maximum length.
    #     """
    #     if len(text) > max_length:
    #         return text[:max_length] + "..."
    #     return text

class Guardrails:
    """
    Orchestrates different guardrail checks.
    """
    def __init__(self): # Removed max_input_length, max_output_length
        self.moderator = ContentModerator()
        # self.max_input_length = max_input_length # Removed
        # self.max_output_length = max_output_length # Removed

    def apply_input_guardrails(self, user_prompt: str) -> Tuple[bool, Optional[str], str]:
        """
        Applies guardrails to the user's input prompt.
        Returns (is_flagged, moderation_reason, moderated_prompt).
        """
        # 1. Content moderation
        is_flagged, reason = self.moderator.moderate_content(user_prompt) # Use user_prompt directly
        if is_flagged:
            return True, f"input {reason}", user_prompt # Return original user_prompt if flagged

        return False, None, user_prompt # Return original user_prompt if not flagged

    def apply_output_guardrails(self, llm_response: str) -> Tuple[bool, Optional[str], str]:
        """
        Applies guardrails to the LLM's generated response.
        Returns (is_flagged, moderation_reason, moderated_response).
        """
        # 1. Content moderation
        is_flagged, reason = self.moderator.moderate_content(llm_response)
        if is_flagged:
            # Replace the response with a generic message if sensitive content is detected.
            return True, f"output {reason}", "I'm sorry, but I cannot provide a response that contains inappropriate content. Please rephrase your query."

        return False, None, llm_response
