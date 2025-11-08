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
# url: https://leetcode.com/problems/build-array-from-permutation/

# (solution and/or notes from LeetCode Solutions tab and/or an AI model)
# this is space complexity o(1) as it permutes the list in-place
# the key insight is to realize that to achieve o(1) space, you must not overwrite
# the original value until you're ready; you have to encode both the original value
# and the new value in one "container" that is the single integer
class Solution:
    def buildArray(self, nums):
        n = len(nums)
        for i in range(n):
            nums[i] = nums[i] + (nums[nums[i]] % n) * n
        for i in range(n):
            nums[i] = nums[i] // n

        return nums

# (my solution)
# time: o(n)
# space: o(n) -- this includes the list comprehension version
class Solution:
    def buildArray(self, nums):
        ans = []
        for i in range(len(nums)):
            ans.append(nums[nums[i]])
        return ans

# ---------------------------------------------------------------------------