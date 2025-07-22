# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# (notes from LeetCode Solutions tab and/or ChatGPT)


# (my solution)
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/score-of-a-string/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# This is a little more elegant because we don't have to do the if statement versus
# my enumerate() version
class Solution:
    def scoreOfString(self, s: str) -> int:

        score = 0
        for i in range(len(s)-1):
            score += abs(ord(s[i]) - ord(s[i+1]))

        return score

# (my solution)
# time: o(n) (correct)
# space: o(1) (correct)
class Solution:
    def scoreOfString(self, s: str) -> int:
        score = 0

        for i, character in enumerate(s):
            if i + 1 < len(s):
                score += abs(ord(s[i]) - ord(s[i + 1]))

        return score
# ---------------------------------------------------------------------------
