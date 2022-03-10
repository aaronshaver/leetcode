
# Discuss tab solution
# author claims time & space: O(w * k), where w = average size of word
# similar to mine, but terser and less space
# I like the early return and no output var / string building
def truncateSentence(self, s: str, k: int) -> str:
    for i, c in enumerate(s):
        if c == ' ':
            k -= 1
            if k == 0:
                return s[: i]
    return s


# my solution
# time O(k*d) where d is avg word length???; worst case O(n)???
# space same as time???
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        output = ''
        for char in s:
            if char != ' ':
                output += char
            else:
                k -= 1
                output += ' '
                if k == 0:
                    break

        return output.rstrip()


# https://leetcode.com/problems/maximum-product-difference-between-two-pairs/

# Discuss tab solution; I was doing more work than necessary by not just doing -1, -2 getter
def maxProductDifference(self, nums: List[int]) -> int:
        nums.sort()
        return nums[-1]*nums[-2]-nums[0]*nums[1]

# my solution; O(n log n) time?; O(1) space?
class Solution:
    def maxProductDifference(self, nums: List[int]) -> int:
        nums.sort()
        return abs((nums[0] * nums[1]) - (nums[-1:].pop() * nums[-2:-1].pop()))


# https://leetcode.com/problems/count-the-number-of-consistent-strings/

# Discuss tab solution; set is implemented as hashset under the hood in Python and has
# lookup time of O(1); as well, this person does a very clever negative counting of
# failed words and substracting of total word count
# time complexity is O(nd), where d is average word length
class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        allowed = set(allowed)
        count = 0

        for word in words:
            for letter in word:
                if letter not in allowed:
                    count += 1
                    break

        return len(words) - count

# my solution; I did have an intuition to make a hashset, just didn't fully follow
# through or think about how it would work
# I believe my time complexity is O(n^2) because of the O(n) char in allowed lookup
class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:

        count = 0
        for word in words:
            flag = True
            for char in word:
                if char not in allowed:  # time save from hashset solution above would be here
                    flag = False
                    break

            if flag:
                count += 1

        return count


# https://leetcode.com/problems/cells-in-a-range-on-an-excel-sheet/

# solution from Discuss tab
# I had thought time complexity was O(n), but I was wrong
# author says: "Time: O((c2 - c1 + 1) * (r2 - r1 + 1)), space: O(1)"
def cellsInRange(self, s: str) -> List[str]:
    c1, c2 = ord(s[0]), ord(s[3])
    r1, r2 = int(s[1]), int(s[4])
    return [chr(c) + str(r) for c in range(c1, c2 + 1) for r in range(r1, r2 + 1)]

# my solution; I did think about doing the "get integer value of letter" thing,
# I just forgot the syntax; in retrospect I should have followed up on this
# intuition
class Solution:
    def cellsInRange(self, s: str) -> List[str]:
        row_min = int(s[1])
        row_max = int(s[4])
        col_min = s[0]
        col_max = s[3]

        all_cols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        index_col_min = all_cols.index(col_min)
        index_col_max = all_cols.index(col_max)

        output = []
        for col in all_cols[index_col_min:index_col_max + 1]:
            for row in range(row_min, row_max + 1):
                output.append(col + str(row))

        return output


# https://leetcode.com/problems/rings-and-rods/

# ultra clever solution from Discussions tab
class Solution:
    def countPoints(self, rings: str) -> int:
        return sum(all(color + rod in rings for color in 'RGB') for rod in '0123456789')

# my solution
# I did try to use all()! But I couldn't figure out the syntax well enough,
# but I'm glad I at least had the intuition to try it
class Solution:
    def countPoints(self, rings: str) -> int:

        parsed_rings = {}
        for i in range(0, len(rings), 2):
            color = rings[i]
            rod = rings[i+1]

            if rod in parsed_rings.keys():
                parsed_rings[rod] = parsed_rings[rod] + color
            else:
                parsed_rings[rod] = color

        count = 0
        for key in parsed_rings.keys():
                rod = parsed_rings[key]
                if 'R' in rod and 'G' in rod and 'B' in rod:
                    count += 1

        return count


# https://leetcode.com/problems/minimum-sum-of-four-digit-number-after-splitting-digits/

# ultra short solution from discussion page based on clever mathy stuff
# I suspected sorting would help, but I wasn't able to properly figure out how
# a little more time on "test cases" and noticing patterns therein may have helped me find this solution myself

class Solution:
    def minimumSum(self, num: int) -> int:
        a,b,c,d=sorted(str(num))
        return int(a+c) + int(b+d)

# my solution:

class Solution:
    # note: I deduced 1,3 and 3,1 pairs would not be needed by
    # testing cases "on paper"

    def check_minimum(self, minimum, first: list[int], second: list[int]):
        total_first = int(first[0] + first[1])
        total_second = int(second[0] + second[1])
        grand_total = total_first + total_second

        return grand_total if grand_total < minimum else minimum

    def minimumSum(self, num: int) -> int:
        digits = []
        for char in str(num):
            digits.append(char)

        # generate all unique 2-digit pairs
        pairs = []
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                pairs.append([digits[i], digits[j]])

        # generate all valid pairs of pairs (i.e. use every original digit exactly once)
        # by grabbing first and last pairs repeatedly
        all_pairs = []
        for i in range(len(pairs) // 2):
            all_pairs.append([pairs[0], pairs[-1:][0]])  # -1: returns a list
            del pairs[0]
            del pairs[-1:]

        minimum = 100000

        for pair_of_pairs in all_pairs:
            first = pair_of_pairs[0]
            second = pair_of_pairs[1]

            minimum = self.check_minimum(minimum, first, second)
            minimum = self.check_minimum(minimum, [first[1], first[0]], second)
            minimum = self.check_minimum(minimum, [first[1], first[0]], [second[1], second[0]])
            minimum = self.check_minimum(minimum, first, [second[1], second[0]])

        return minimum


# https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/submissions/

# my solution
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        return ''.join(word1) == ''.join(word2)

# my gut told me this was O(n) and that it could be better, but I struggled with how to
# effectively do one char at a time
# browsing the discussions yielded (get it!?) this super clever yet understandable solution
# time complexity is O(min(m,n)) -- exits at the first char pair mismatch
# space complexity is O(1) since nothing is stored
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        for c1, c2 in zip(self.generate(word1), self.generate(word2)):
            if c1 != c2:
                return False
        return True

    def generate(self, wordlist: List[str]):
        for word in wordlist:
            for char in word:
                yield char
        yield None


# https://leetcode.com/problems/count-of-matches-in-tournament/submissions/

class Solution:
    def numberOfMatches(self, n: int) -> int:
        total_matches = 0
        while n > 1:
            in_match = n if n % 2 == 0 else n - 1
            matches_this_round = in_match // 2
            total_matches += matches_this_round
            n -= matches_this_round

        return total_matches

# ^ re: above -- after looking at discussion page: the answer is simply n - 1; no algorithm needed
# in retrospect, next time I could work through more sample inputs and see if I notice
# a pattern/answer that keeps recurring; I got impatient and jumped ahead to code


# https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        binary_values = []
        current_node = head
        while True:
            binary_values.append(current_node.val)
            if current_node.next == None:
                break
            current_node = current_node.next

        decimal = 0
        length = len(binary_values)
        for i in range(length):
            decimal += binary_values[i] * (2 ** (length - 1 - i))

        return decimal


# https://leetcode.com/problems/sum-of-all-odd-length-subarrays/submissions/

class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        length = len(arr)
        if length % 2 != 0:
            current_odd = length
        else:
            current_odd = length - 1

        total = 0
        while current_odd >= 1:
            start = 0
            end = start + current_odd
            while end <= length:
                total += sum(arr[start:end])
                start += 1
                end += 1
            current_odd -= 2

        return total


# https://leetcode.com/problems/minimum-number-of-moves-to-seat-everyone/

class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:

        seats = sorted(seats)
        students = sorted(students)

        total_moves = 0
        for student in students:
            if student == seats[0]:  # already correct position; no move needed
                del seats[0]
                continue
            distance = abs(student - seats[0])
            total_moves += distance
            del seats[0]

        return total_moves


# https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/

class Solution:
    def maxDepth(self, s: str) -> int:
        opens = 0
        highest_opens = 0
        stripped = ''.join(char for char in s if char not in '0123456789-+*/')

        for token in stripped:
            if token == '(':
                opens += 1
                if opens > highest_opens:
                    highest_opens = opens
            elif token == ')':
                opens -= 1

        return highest_opens


# https://leetcode.com/problems/find-target-indices-after-sorting-array/submissions/

# my solution
class Solution:
    def targetIndices(self, nums: List[int], target: int) -> List[int]:
        sorted_nums = sorted(nums)
        output = []
        for i in range(len(nums)):
            if sorted_nums[i] == target:
                output.append(i)

        return output

# clever solution I found in the discussions that skips needing to sort
# by figuring out the point at which less-than-target numbers start
class Solution:
    def targetIndices(self, nums, target):
        lt_count, eq_count = 0, 0
        for n in nums:
            if n < target:
                lt_count += 1
            elif n == target:
                eq_count += 1

        return list(range(lt_count, lt_count+eq_count))


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
