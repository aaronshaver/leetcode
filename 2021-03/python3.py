
# -----------------------------------------------------------------------------
# https://leetcode.com/problems/number-of-good-pairs/submissions/
class Solution:
    def numIdenticalPairs(self, nums):
        indices = {}
        
        for i, num in enumerate(nums):
            if num not in indices.keys():
                indices[num] = [i]
            else:
                indices[num] = indices[num] + [i]
        
        count = 0
        for key in indices.keys():
            length = len(indices[key])
            count += (length - 1) * length // 2  # one less than triangular number
        
        return count

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/richest-customer-wealth/submissions/
class Solution:
    def maximumWealth(self, accounts):
        highest = 0
        for i in range(len(accounts)):
            wealth = sum(accounts[i])
            if wealth > highest:
                highest = wealth
        
        return highest

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/submissions/
class Solution:
    def kidsWithCandies(self, candies, extraCandies):
        output = []
        highest_candy = max(candies)
        
        for c in candies:
            if (c >= highest_candy) or (c + extraCandies >= highest_candy):
                output.append(True)
            else:
                output.append(False)
        
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/shuffle-the-array/submissions/
class Solution:
    def shuffle(self, nums, n):
        output = []
        for i in range(n):
            output.append(nums[i])
            output.append(nums[i+n])
        
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/defanging-an-ip-address/submissions/
class Solution:
    def defangIPaddr(self, address):
        return address.replace('.', '[.]')

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/running-sum-of-1d-array/submissions/
class Solution:
    def runningSum(self, nums):
        running = 0
        output = []
        for i in range(len(nums)):
            running += nums[i]
            output.append(running)
        return output

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