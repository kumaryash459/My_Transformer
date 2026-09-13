import re


class BasicTokenizer:

    def tokenize(self, text: str) -> list[str]:

        text = text.lower().strip()

        # Separate punctuation
        text = re.sub(
            r"([.,!?;:])",
            r" \1 ",
            text
        )

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.split()