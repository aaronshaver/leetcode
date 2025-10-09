# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# (solution and/or notes from LeetCode Solutions tab and/or an AI model)


# (my solution)
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/divisible-and-non-divisible-sums-difference/

# (solution and/or notes from LeetCode Solutions tab and/or an AI model)


# (my solution)
# time: o(n)
# space: o(1)
class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        num1 = 0
        num2 = 0
        for number in range(1, n + 1):
            if number % m != 0:
                num1 += number
            if number % m == 0:
                num2 += number
        return num1 - num2
# ---------------------------------------------------------------------------

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
