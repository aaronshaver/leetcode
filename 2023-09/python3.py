# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# notes from LeetCode Solutions tab and/or ChatGPT


# my solution
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/can-place-flowers/?envType=study-plan-v2&envId=leetcode-75

# notes from LeetCode Solutions tab and/or ChatGPT
# solution from mulliganaceous on leetcode that's quite terse:
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        count = 1
        beds = 0
        for bed in flowerbed:
            if bed:
                beds += (count - 1) // 2
                count = 0
            else:
                count += 1

        beds += count // 2

        return beds >= n

# my solution
# time: O(n)
# space: O(1)
class Solution:
    def getCapacity(self, capacity, counter):
        if counter == 2:
            return capacity + 1
        else:
            return capacity + ceil(counter / 2)

    def canPlaceFlowers(self, flowerBed: List[int], n: int) -> bool:
        capacity = 0
        counter = 0
        flowersSeen = 0

        # special case of [0]
        if len(flowerBed) == 1 and flowerBed[0] == 0:
            return True

        for i, plot in enumerate(flowerBed):
            if not plot: # empty space
                counter += 1
            else:
                flowersSeen += 1
                if flowersSeen == 1:
                    if counter != 1:
                        capacity = self.getCapacity(capacity, counter)
                elif counter != 0:
                    capacity += ceil(counter / 2 - 1)
                counter = 0

        # end zeroes
        if counter > 1:
            capacity = self.getCapacity(capacity, counter)

        return capacity >= n
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/?envType=study-plan-v2&envId=leetcode-75

# notes from LeetCode Solutions tab and/or ChatGPT
# someone on the site noticed you can get O(1) space complexity by modifying the list
# after doing the comparison, e.g.:
for idx, candy in enumerate(candies):
    canHaveMaxCandies = (candy + extraCandies) >= maxNumCandies
    candies[idx] = canHaveMaxCandies

# my solution
# time: O(n)
# space: O(n)
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        largest = max(candies)
        has_greatest = []
        for item in candies:
            has_greatest.append(item + extraCandies >= largest)
        return has_greatest
# ---------------------------------------------------------------------------

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
# I had to get a hint; went down a wrong rabbithole with .count()
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