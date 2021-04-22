
# -----------------------------------------------------------------------------
# https://leetcode.com/problems/shuffle-string/submissions/
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        # this is (maybe?) intentionally inefficient: I accidently got a peek
        # at a solution and so I tried to do it in a different way, although 
        # I think this isn't bad. It's not exponential; it's basically always
        # looping through n 3 times so the growth rate should be linear
        output = ""
        pairs = zip([letter for letter in s], indices)
        
        for tuple in sorted(pairs, key=lambda x:x[1]):  # sort by second item in tuple
            output += tuple[0]
            
        return output
        
# -----------------------------------------------------------------------------
# https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(nums)
        
        counts = []
        for num in nums:
            counts.append(sorted_nums.index(num))
        
        return counts
            
# my original, naive algorithm that's N^2:
#
#    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
#        counts = []
#        for i, num in enumerate(nums):
#            count = 0
#            for j in range(len(nums)):
#                if j == i:
#                   continue
#                if nums[j] < nums[i]:
#                    count += 1
#            counts.append(count)
#        
#        return counts

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/check-if-the-sentence-is-pangram/submissions/
class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        counts = {}
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            counts[letter] = 0
            
        for char in sentence:
            counts[char] = counts[char] + 1
        
        flag = True
        for count in counts.values():
            if count == 0:
                return False

        return True

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/design-parking-system/submissions/
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.carTypes = {}
        self.carTypes[3] = small
        self.carTypes[2] = medium
        self.carTypes[1] = big

    def addCar(self, carType: int) -> bool:
        if self.carTypes[carType] < 1:
            return False
        else:
            self.carTypes[carType] = self.carTypes[carType] - 1
            return True

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/jewels-and-stones/submissions/
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        counter = 0
        for stone in stones:
            if stone in jewels:
                counter += 1
        return counter

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