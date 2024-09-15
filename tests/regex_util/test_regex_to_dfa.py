from pyformlang.regular_expression import Regex
import pytest
import random
import itertools
from project.regex_util import regex_to_dfa

REGEXES = [
    "a* b o b a*",
    "[a1 a2 a3 a4]*",
    "([A T G C][A T G C][A T G C])*",
    "[I WE YOU][LOVE HATE DESTROY][PEOPLES WORLD WATER HOME]",
]


class TestRegexToDfa:
    @pytest.mark.parametrize("regex_str", REGEXES)
    def test(self, regex_str: str) -> None:
        regex = Regex(regex_str)
        regex_cfg = regex.to_cfg()
        regex_words = regex_cfg.get_words()

        dfa = regex_to_dfa(regex_str)
        minimized_dfa = dfa.minimize()
        assert dfa.is_deterministic()
        assert dfa.is_equivalent_to(minimized_dfa)

        for _ in range(1):
            if regex_cfg.is_finite():
                all_word_parts = list(regex_words)
                word_parts = random.choice(all_word_parts)
            else:
                index = random.randint(0, 2**9)
                word_parts = next(itertools.islice(regex_words, index, None))

            word = map(lambda x: x.value, word_parts)

            assert dfa.accepts(word)
