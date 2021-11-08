
# -----------------------------------------------------------------------------
# https://leetcode.com/problems/decompress-run-length-encoded-list/submissions/
class Solution:
    def decompressRLElist(self, nums: List[int]) -> List[int]:
        output = []
        for i in range(0, len(nums), 2):
            index_of_char = i + 1
            count_of_char = nums[i] 
            output = output + [nums[index_of_char] for _ in range(count_of_char)]
            
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/subtract-the-product-and-sum-of-digits-of-an-integer/submissions/

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        digits = []
        remaining = n
        while True:
            next_num = remaining % 10 
            remaining = remaining // 10
            digits.append(next_num)
            if remaining == 0:
                break

        # as of Python 3.8, there's a math.prod() but I wanted to do it without imports
        product = 1
        for elem in digits:
            product = product * elem
            
        return product - sum(digits)

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/decode-xored-array/submissions/

class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        output = [first]
        left = first
        for num in encoded:
            new = left ^ num  # xor next encoded num with last known unencoded
            output.append(new)
            left = new  # new last known unencoded
            
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/build-array-from-permutation/submissions/

class Solution:
    def buildArray(self, nums):
        output = []
        for i in range(len(nums)):
            output.append(nums[nums[i]])
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/concatenation-of-array/submissions/

class Solution:
    def getConcatenation(self, nums):
        return nums * 2

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/final-value-of-variable-after-performing-operations/submissions/

class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        x = 0
        for op in operations:
            if op in ["X++", "++X"]:
                x += 1
            else:
                x -= 1
        return x
