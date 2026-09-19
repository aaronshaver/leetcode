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
# url: https://leetcode.com/problems/two-sum/description/

# (solution and notes from LeetCode Solutions tab and/or AI model)
# I half-remembered the solution but I got a hint from the LLM to get to the
# time complexity of O(n)



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
