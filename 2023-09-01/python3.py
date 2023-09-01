# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# discuss tab solution


# my solution
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# url: https://leetcode.com/problems/merge-strings-alternately/?envType=study-plan-v2&envId=leetcode-75

# discuss tab solution


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