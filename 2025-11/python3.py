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


# (my solution)
# time:
# space:
class Solution:
    def buildArray(self, nums):
        ans = []
        for i in range(len(nums)):
            ans.append(nums[nums[i]])
        return ans

# ---------------------------------------------------------------------------