import re
import string

class ChallengeVerifier:
    def __init__(self):
        # A simple number word to digit mapping for deterministic normalization
        self.number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10",
            "to": "2", "too": "2", "for": "4",
            "oh": "0", "o": "0", "won": "1", "tree": "3", "free": "3", "ate": "8"
        }
        # Known phonetic misrecognitions of 'present sir' in Whisper
        self.salutation_variants = {
            "presentsir", "presenter", "resonator", "resonsor",
            "presentyear", "presencesir", "presence", "present", "sir"
        }
        
    def normalize_text(self, text: str) -> str:
        """
        Deterministically normalizes text:
        - Lowercases
        - Removes punctuation
        - Replaces number words with digits
        - Removes all whitespace to handle Whisper digit grouping (e.g. "214" vs "2 1 4")
        """
        if not text:
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens
        tokens = text.split()
        
        # Replace numbers
        normalized_tokens = [self.number_map.get(t, t) for t in tokens]
        
        # Join back and remove spaces
        return "".join(normalized_tokens)

    def _strip_salutation(self, text: str) -> str:
        for s in sorted(self.salutation_variants, key=len, reverse=True):
            if text.startswith(s):
                return text[len(s):]
        return text

    def verify_challenge(self, expected: str, transcribed: str) -> bool:
        """
        Verifies if the transcribed text matches the expected challenge phrase.
        Supports exact match, salutation prefix stripping, and dynamic digit sequence matching.
        """
        norm_expected = self.normalize_text(expected)
        norm_transcribed = self.normalize_text(transcribed)
        
        result = False
        if norm_expected and norm_transcribed:
            # 1. Exact normalized equality
            if norm_expected == norm_transcribed:
                result = True
            else:
                # 2. Stripped salutation equality (e.g. presentsir512 vs resonator512 or just 512)
                stripped_exp = self._strip_salutation(norm_expected)
                stripped_trans = self._strip_salutation(norm_transcribed)
                if stripped_exp and stripped_trans and stripped_exp == stripped_trans:
                    result = True
                else:
                    # 3. Dynamic challenge digit verification (entropy check)
                    digits_exp = "".join(re.findall(r"\d", norm_expected))
                    digits_trans = "".join(re.findall(r"\d", norm_transcribed))
                    if digits_exp and digits_trans and digits_exp == digits_trans:
                        result = True
        
        # Temporary development-only diagnostic logging
        import logging
        logger = logging.getLogger("challenge_verifier")
        if not logger.handlers:
            import sys
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        logger.info("\n--- CHALLENGE VERIFICATION DIAGNOSTICS ---")
        logger.info(f"EXPECTED RAW:\n{expected}")
        logger.info(f"ASR RAW:\n{transcribed}")
        logger.info(f"EXPECTED NORMALIZED:\n{norm_expected}")
        logger.info(f"ASR NORMALIZED:\n{norm_transcribed}")
        logger.info(f"CHALLENGE RESULT:\n{'PASS' if result else 'FAIL'}")
        logger.info("------------------------------------------\n")
        
        return result
