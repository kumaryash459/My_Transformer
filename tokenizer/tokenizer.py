import re


class BasicTokenizer:

    def tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Separate punctuation
        text = re.sub(
            r'([.,!?;:"»«()\-])',
            r" \1 ",
            text
        )

        # Separate apostrophes
        text = re.sub(
            r"(['])",
            r" \1 ",
            text
        )

        # Normalize whitespace again
        text = re.sub(r"\s+", " ", text)

        return text.split()