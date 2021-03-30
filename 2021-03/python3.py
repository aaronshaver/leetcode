

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/two-sum/submissions/
class Solution:
    def twoSum(self, nums, target):
        nums_set = set(nums)
        for i in range(len(nums) - 1):
            current_num = nums[i]
            pair_num = target - current_num
            if pair_num in nums_set:
                if pair_num != current_num:
                    return(i,nums.index(pair_num))
                elif nums.count(pair_num) == 2:
                    for j in range(i+1,len(nums)):
                        if nums[j] == pair_num:
                            return [i, j]