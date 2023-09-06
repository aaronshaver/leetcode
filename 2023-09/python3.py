# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# discuss tab solution


# my solution
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/greatest-common-divisor-of-strings/?envType=study-plan-v2&envId=leetcode-75

# discuss tab solution
# this is actually GPT-4's solution; quite clever:
# time -- I was incorrect:
# O(n*m) because we loop through both strings once; "The string
# multiplication and comparison take time proportional to the lengths of both strings."
#
# space -- I was correct; GPT-4 wants to use this notation:, O(min(n, m))
# more commonly you'd probably just have O(n) and note that n is the shorter string
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        for i in range(min(len(str1), len(str2)), 0, -1):
            chunk = str1[:i]
            if str1[:i] == str2[:i]:
                if str1.replace(chunk, "") == "" and str2.replace(chunk, "") == "":
                    return chunk
        return ""

# my solution
# time: O(n)?
# space: O(m) where m is the shorter of the two strings??
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        size = 0
        candidate = ""
        while True:
            size += 1
            if size > len(str2) or size > len(str1):
                return candidate
            chunk = str1[0:size]
            if chunk * (len(str1) // size) == str1 and \
            chunk * (len(str2) // size) == str2:
                candidate = chunk
        return candidate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/merge-strings-alternately/?envType=study-plan-v2&envId=leetcode-75

# discuss tab solution
# near as I can tell, there's no way to improve the time or space complexity of
# O(n)
#
# there was this "clever" one-liner in the Discussion tab (below); it's actually
# not that bad in terms of understandability given it's so terse; a key to
# understanding it is: The zip function stops pairing when the shortest input
# iterable is exhausted.
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return "".join(a + b for a, b in zip(word1, word2)) + word1[len(word2):] + word2[len(word1):]

# my solution
# I don't believe there is a way to get below O(n) with either time or space
# I thought about doing a dual pointers thing, but the immutability of strings
# in Python makes it difficult to do any kind of in-place swapping
# time: O(n)
# space: O(n)
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        shortest_length = len(word1) if len(word1) <= len(word2) else len(word2)
        output = []
        for i in range(shortest_length):
            output += word1[i]
            output += word2[i]

        if (len(word1) != len(word2)):
            longest_word = word1 if len(word1) > len(word2) else word2
            output += longest_word[shortest_length:]
        return "".join(output)
# ---------------------------------------------------------------------------