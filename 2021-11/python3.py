
# -----------------------------------------------------------------------------
# https://leetcode.com/problems/split-a-string-in-balanced-strings/submissions/

class Solution:
    def balancedStringSplit(self, s: str) -> int:
        subsets_count = 0 
        
        buffer = ''
        for char in s:
            buffer  += char
            
            if buffer.count('L') == buffer.count('R'):
                subsets_count += 1
        
        return subsets_count

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/goal-parser-interpretation/submissions/

class Solution:
    def interpret(self, command: str) -> str:
        output = ''
        i = 0
        while i < len(command):
            if command[i] == 'G':
                output += 'G'
                i += 1
            elif command[i + 1] == ')':
                output += 'o'
                i += 2
            elif command[i + 1] == 'a':
                output += 'al'
                i += 4
        
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/create-target-array-in-the-given-order/submissions/

class Solution:
    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        target = []
        for i in range(len(nums)):
            if index[i] == len(target):
                target.append(nums[i])
            else:
                target.insert(index[i], nums[i])
        
        return target

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/submissions/

class Solution:
    def numberOfSteps(self, num: int) -> int:
        if num == 0:
            return 0
        count = 0
        while True:
            if num % 2 == 0:
                num = num / 2
            else:
                num -= 1
            count += 1
            if num == 0:
                break
        
        return count

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
