# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# (solution and notes from LeetCode Solutions tab and/or AI model)


# (my solution)
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/valid-parentheses/

# (solution and notes from LeetCode Solutions tab and/or AI model)
#
# after a couple small hints, like using a list as a stack, and realizing
# slicing off the last two entries in the list is O(n)
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) < 2 or s[0] in [']', '}', ')']:
            return False

        parens = []
        parensMap = { ')': '(', ']': '[', '}': '{'}
        for char in s:
            parens.append(char)
            if len(parens) > 1 and char in [']', '}', ')']:
                if parens[-2] == parensMap[char]:  # if two back is opening paren
                    parens.pop()  # remove the closed pair
                    parens.pop()  # remove the closed pair
        return not parens  # every pair closed and remove
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) < 2 or s[0] in [']', '}', ')']:
            return False

        parens = []
        parensMap = { ')': '(', ']': '[', '}': '{'}
        for char in s:
            parens.append(char)
            if len(parens) > 1 and char in [']', '}', ')']:
                if parens[-2] == parensMap[char]:  # if two back is opening paren
                    parens = parens[:-2]  # remove the closed pair
        return not parens  # every pair closed and removed


# (my solution)
# time: O(n^2) because string appends can add up
# space: O(n)
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) < 2 or s[0] in [']', '}', ')']:
            return False

        parens = ""
        parensMap = { ')': '(', ']': '[', '}': '{'}
        for char in s:
            parens += char
            if len(parens) > 1 and char in [']', '}', ')']:
                if parens[-2] == parensMap[char]:  # if two back is opening paren
                    parens = parens[:-2]  # remove the closed pair
        return not parens  # every pair closed and removed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/two-sum/description/

# (solution and notes from LeetCode Solutions tab and/or AI model)
#
# I half-remembered the solution but I needed a hint from the LLM to get the
# hint about the seen dictionary
#
# My new version which is still O(n), since constant factors (2n) are dropped:
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}
        for i, number in enumerate(nums):
            if number in seen:
                seen[number] = seen[number] + [i]
            else:
                seen[number] = [i]
        for key in seen.keys():
            difference = target - key
            if difference in seen:
                if key == difference:
                    if len(seen[difference]) > 1:
                        return seen[difference]
                else:
                    return [seen[key][0], seen[difference][0]]
# nicer version from LLM that avoids the two passes:
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}

        for i, number in enumerate(nums):
            difference = target - number

            if difference in seen:
                return [seen[difference], i]

            seen[number] = i

# (my solution)
# time: O(n^2)
# space: O(n)
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        i = 0
        for first in nums:
            j = i + 1
            for second in nums[i + 1:]:
                if first + second == target:
                    return [i, j]
                j += 1
            i += 1

# ---------------------------------------------------------------------------
