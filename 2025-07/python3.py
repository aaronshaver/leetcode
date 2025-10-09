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
# This solution using mathematical forumulas to do time complexity o(1) instead
# of o(n)
class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Calculate the sum of all numbers from 1 to n.
        # The formula for the sum of the first n integers is n * (n + 1) / 2.
        total_sum = n * (n + 1) // 2

        # Find the number of integers from 1 to n that are divisible by m.
        # This is equivalent to floor(n / m).
        k = n // m

        # Calculate the sum of numbers divisible by m (num2).
        # This is an arithmetic series: m + 2m + ... + km = m * (1 + 2 + ... + k).
        # We use the sum formula again for the series 1 to k.
        sum_of_multiples = m * (k * (k + 1) // 2)

        # The final result is total_sum - 2 * sum_of_multiples.
        # This comes from the identity: num1 - num2 = (total_sum - num2) - num2.
        return total_sum - 2 * sum_of_multiples

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
