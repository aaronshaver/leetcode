

# https://leetcode.com/problems/count-number-of-pairs-with-absolute-difference-k/submissions/

# my solution, which is slow at O(n^2)
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range (i + 1, len(nums)):
                if abs(nums[i] - nums[j]) == k:
                    count += 1
        
        return count

# slick, fast, and easy to read solution I saw in the Discussions area:
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        d = {}
        count = 0
        
        for num in nums:
            d[num] = d.get(num, 0) + 1
            count += d.get(num - k, 0)
            count += d.get(num + k, 0)
                
        return count


# https://leetcode.com/problems/find-center-of-star-graph/submissions/

class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        # return the first node value seen more than once
        # this works because of the specific shape of this kind of graph
        node_values = {}
        for edge in edges:
            for value in edge:
                if value in node_values:
                    return value
                else:
                    node_values[value] = 1


# https://leetcode.com/problems/sorting-the-sentence/

from operator import itemgetter

class Solution:
    def sortSentence(self, s: str) -> str:
        raw_words = s.split()
        tuples = []
        for raw_word in raw_words:
            # extract index, word into a tuple
            tuples.append((raw_word[-1:], raw_word[:-1]))

        sorted_tuples = sorted(tuples, key=itemgetter(0))

        output = ""
        for tuple in sorted_tuples:
            output = output + tuple[1] + ' '
    
        return output.rstrip()


# https://leetcode.com/problems/maximum-number-of-words-found-in-sentences/submissions/

class Solution:
    def mostWordsFound(self, sentences: List[str]) -> int:
        longest = 0
        for sentence in sentences:
            length = len(sentence.split())
            if length > longest:
                longest = length
        
        return longest


# -----------------------------------------------------------------------------
# https://leetcode.com/problems/xor-operation-in-an-array/submissions/

class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        if n == 1:
           return start
    
        nums = []
        output = None
        for i in range(n):
            nums.append(start + 2 * i)
            if len(nums) == 1:
                left_num = nums[0]
            elif len(nums) > 1:
                output = left_num ^ nums[-1]
                left_num = output
                
        return output

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/range-sum-of-bst/submissions/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def __init__(self):
        self.total = 0  # because of goofy LeetCode shared instance
    
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if root.val <= high and root.val >= low:
            self.total += root.val
        if root.left:
            self.rangeSumBST(root.left, low, high)
        if root.right:
            self.rangeSumBST(root.right, low, high)
        
        return self.total

# -----------------------------------------------------------------------------
# https://leetcode.com/problems/count-items-matching-a-rule/submissions/

class Solution:
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        
        if ruleKey == "type":
            return [x[0] for x in items].count(ruleValue)
        if ruleKey == "color":
            return [x[1] for x in items].count(ruleValue)
        if ruleKey == "name":
            return [x[2] for x in items].count(ruleValue)

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
