

# https://leetcode.com/problems/decrypt-string-from-alphabet-to-integer-mapping/

# Discuss tab solution
# this is obnoxiously good, haha
# I would say I need to get better at regex; however, I find I seldom use it at jobs
# and it can sometimes become "write-only code", i.e. easy to write, hard to read
def freqAlphabets(self, s):
    return ''.join(chr(int(i[:2]) + 96) for i in re.findall(r'\d\d#|\d', s))

# my solution
# time: O(n); space: O(n)
# should probably extract out the chr(int() stuff into a function
class Solution:
    def freqAlphabets(self, s: str) -> str:
        output = ''
        i = 0
        offset = 96
        while i < len(s):
            current_char = s[i]
            if (i + 2) < len(s):
                if s[i + 2] == '#':  # j char or higher
                    output += chr(int(current_char + s[i + 1]) + offset)
                    i += 2
                else:
                    output += chr(int(current_char) + offset)  # a-i
            else:
                output += chr(int(current_char) + offset)  # a-i
            i += 1

        return output


# https://leetcode.com/problems/minimum-bit-flips-to-convert-number/

# Discuss tab
# well this is humbling... >_<
# it makes sense, too. XOR will show all the spots where only one of the two
# binary numbers is 1, thus implicitly adding those spots up == number of flips needed
# to reach an AND state... clever
def minBitFlips(self, start: int, goal: int) -> int:
    return (start ^ goal).bit_count()

# my solution
# time: O(n) where n is number of binary digits in longest of the two nums
# space: effectively O(1)
#
# this works by making a bit mask using left shift 1 << i, e.g. 1 << 3 == a mask of 1000
# then bitwise AND with the num, gives us a binary num that indicates whether both mask
# and num have a 1 in that digit; e.g. 10 & (1<<2) == 000, and 7 & (1<<2) == 100
# then bit shift right chops off the trailing 0s
# so in this case we'd have 0 for the start, and 1 for the goal, meaning digit at that
# place has to change for the start num, and thus our count of operations has to increase
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        length = max(len(bin(start)[2:]), len(bin(goal)[2:]))  # '2:'' chops off 0b
        count = 0
        for i in range(length):
            if ((start & (1 << i)) >> i) != ((goal & (1 << i)) >> i):
                count += 1

        return count


# https://leetcode.com/problems/sum-of-all-subset-xor-totals/

# Disuss tab solutions
# pasting links instead of solutions here
# there's a few different ways: recursive, iterative, and math-trick
# I'll read through these after lunch!
# https://leetcode.com/problems/sum-of-all-subset-xor-totals/discuss/1211213/Python-Bitwise-OR-with-explanation-O(n)
# https://leetcode.com/problems/sum-of-all-subset-xor-totals/discuss/1230324/Python-all-the-4-different-ways-to-solve-this-problem

# my solution
# time: O(n^2); space: O(n^2)
#
# This was a tough one. I spent time reading:
# https://www.geeksforgeeks.org/print-all-possible-combinations-of-r-elements-in-a-given-array-of-size-n/
# but it was a little much, and so I resorted to the library function .combinations()
#
# as an aside, I don't quite understand why single numbers aren't 0;
# number XORed with itself should be 0; problem wants it to be the number itself
import itertools

class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return nums[0]

        singles_total = 0
        for num in nums:
            singles_total += num  # for case of group size == 1

        groups = []
        for group_size in range(2, len(nums) + 1):
            for subset in itertools.combinations(nums, group_size):
                groups.append(subset)

        groups_total = 0
        for group in groups:
            total = 0
            # := is assignment expression, introduced in Python 3.8 in 2019; acts as reducer
            temp = [total := total ^ x for x in group]
            groups_total += total

        return singles_total + groups_total


# Discuss tab solution(s)
# didn't find anything mind-blowing, but this was cute using dictionaries, which I'd briefly considered,
# but it is less space efficient:
class Solution:
    def countGoodRectangles(self, rectangles):
        d = {}
        for i,j in rectangles:
            minimum = min(i,j)
            d[minimum] = d.get(minimum,0)+1
        return d[max(d)]

# my solution
# time: O(n); space: O(1)
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        largest_seen, count = 0, 0
        for rectangle in rectangles:
            shortest_side = min(rectangle)
            if shortest_side > largest_seen:
                largest_seen = shortest_side
                count = 1
            elif shortest_side == largest_seen:
                count += 1
        return count


# https://leetcode.com/problems/maximum-69-number/

# Discuss tab solution
# there were solutions like this that involve string conversion, which I tried to avoid, because
# I've gotten a real inteview question where the interviewer said "now do it without string conversion"
def maximum69Number(self, num):
    return int(str(num).replace('6', '9', 1))

# I was happy to see another person had a solution similar to mine without string conversion;
# theirs is a little better in using an int for the "six index" instead of my list, and it's shorter
class Solution:
    def maximum69Number (self, num: int) -> int:
        i = 0
        tem = num
        sixidx = -1
        while tem > 0:
            if tem % 10 == 6:
                sixidx = i  #refresh sixidx when found 6 at large digit.
            tem = tem//10
            i += 1
        return (num + 3 *(10**sixidx)) if sixidx != -1 else num
# here's a very clean solution using divmod, which I hadn't heard of:
def maximum69Number (self, num: int) -> int:

    six_index = -1
    remainder = num
    pos = 0

    while remainder:
        remainder, digit = divmod(remainder, 10)
        if digit == 6:
            six_index = pos
        pos += 1

    return num + 3 * 10 ** six_index if six_index >= 0 else num

# my solution
# time: O(n); I'm not totally sure; input being 10x bigger won't make the algorithm 10x slower
# but it's also not strictly linear with n?; there's one more iteration for every 10x of n
# space: O(1) except for a small, slow-growing list
class Solution:
    def maximum69Number (self, num: int) -> int:
        six_locations = []  # decimal place within the original num
        for i in range(999):
            output = num // (10 ** i)
            if output == 0:
                break
            if output % 10 == 6:
                six_locations.append(i)
            i += 1

        if len(six_locations) == 0:
            return num  # number was all 6s

        largest_six = max(six_locations)
        return num + (3 * (10 ** largest_six))  # 3 because we're tring to get up to 9


# https://leetcode.com/problems/find-the-highest-altitude/

# Discuss tab solution
def largestAltitude(self, A):
    return max(0, max(accumulate(A)))
# note 1: the reason for the "0, " is because of data sets like [-4,-3,-2,-1,4,3,2]
# where the accumulated addition of those values is -1, yet starting altitude is 0,
# which is higher
#
# note 2: LeetCode seems to have imported already, but you'll need to do:
#     from itertools import accumulate
# or similar to start working with accumulate
# simple example:
#
# > max(accumulate([1,2,3]))
# > 6

# my solution
# time: O(n); space: O(1)
# I did have the intuition that we wanted an accumulator / reducer here
# (as I've used in TypeScript/JavaScript) I simply didn't know the syntax
class Solution:
    def largestAltitude(self, gains: List[int]) -> int:
        highest, current = 0, 0
        for gain in gains:
            current += gain
            if current > highest:
                highest = current

        return highest


# https://leetcode.com/problems/matrix-diagonal-sum/

# Discuss tab solutions
# I saw a lot of terse solutions like the one below, but they aren't any faster,
# and they're IMO much harder to read
def diagonalSum(self, mat: List[List[int]]) -> int:
    n = len(mat)
    return sum(row[r] + row[n - 1 - r] for r, row in enumerate(mat)) - (0, mat[n // 2][n // 2])[n % 2]

# my solution
# time: O(n) where n is the length of a side of the square matrix
# space: effectively O(1)
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        if len(mat) == 1:  # special case of 1x1 matrix
            return mat[0][0]

        total = 0
        j = len(mat) - 1
        for i in range(len(mat)):
            if i == j:  # middle item for odd-size matrices
                total += mat[i][i]
            else:
                total += mat[i][i]
                total += mat[i][j]
            i += 1
            j -= 1

        return total


# https://leetcode.com/problems/minimum-time-visiting-all-points/

# Discuss tab solution
# this cleverly takes advantage of the mathematical truth that the answer is always
# the greater of the two differences (x difference, y difference)
# Also remove the need for a counter
# big O time and space are the same as mine (aside from counter)
def minTimeToVisitAllPoints(self, p: List[List[int]]) -> int:
    return sum(max(abs(p[i][0] - p[i - 1][0]), abs(p[i][1] - p[i - 1][1])) for i in range(1, len(p)))

# my solution
# time: O(n); space: effectively O(1)
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        count = 0
        for i, point in enumerate(points):
            if i + 1 < len(points):
                next_position = points[i + 1]
                x_difference = abs(point[0] - next_position[0])
                y_difference = abs(point[1] - next_position[1])

                num_diagonals = min(x_difference, y_difference)
                count += num_diagonals

                # now add leftover vertical, horizontal movement that cannot
                # be done diagonally; this might be 0 in one or both cases
                count += x_difference - num_diagonals
                count += y_difference - num_diagonals

        return count


# https://leetcode.com/problems/number-of-strings-that-appear-as-substrings-in-word/

# Discuss tab solution
# cleverly removes the need to declare a counter variable
#
# I also need to read up on the KMP Algorithm for Pattern Searching
class Solution:
    def numOfStrings(self, patterns: List[str], word: str) -> int:
        return sum(x in word for x in patterns)

# my solution
# time: O(mn) where m is number of patterns; space: O(1), basically
#
# I thought about comparing the patterns with each other and maybe saving work by excluding
# ones that can't match: if "a" isn't in the word, "ab" will never be
#
# I thought about implementing a "cursor" approach of scanning through 'word' but it
# seemed like my impl would be worse than using built-in contains() which averages
# O(n)
class Solution:
    def numOfStrings(self, patterns: List[str], word: str) -> int:
        count = 0
        for pattern in patterns:
            if pattern in word:
                count += 1

        return count


# https://leetcode.com/problems/flipping-an-image/

# Discuss tab solution
# similar to mine; says time, space are O(mn) where m*n == image size, which makes sense,
# but since the image can be represented as one long list of concatenated rows, it's the
# same as O(n) IMO
#
# some debate on [::-1] vs reversed(), theoretically reversed() is better, BUT:
# 'The time complexity is O(1), but in practice, that rarely matters, because if you are calling
# reversed then you are also iterating over it, which takes at least O(n) time. If you break early
# from the iteration then it will matter'
# https://stackoverflow.com/questions/65540349/time-complexity-of-reversed-in-python-3#comment115874825_65540349
class Solution:
    def flipAndInvertImage(self, A):
        return [[1^q for q in row[::-1]] for row in A]

# my solution
# time: O(n); space: O(n)
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        output = []
        for row in image:
            new_row = []
            for pixel in row[::-1]:
                new_row.append(abs(pixel - 1))
            output.append(new_row)

        return output


# https://leetcode.com/problems/count-equal-and-divisible-pairs-in-an-array/

# Discuss tab solution
# same logic as mine, but I keep forgetting about this handy defaultdict subclass
# it differs in that it won't throw an exception of KeyError if a key doesn't exist, it'll
# just add it when you insert
def countPairs(self, nums: List[int], k: int) -> int:
    cnt, d = 0, defaultdict(list)
    for i, n in enumerate(nums):
        d[n].append(i)
    for indices in d.values():
        for i, a in enumerate(indices):
            for b in indices[: i]:
                if a * b % k == 0:
                    cnt += 1
    return cnt

# my solution
# time: O(n); space: O(n)
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        indices = {}
        for i, num in enumerate(nums):
            if num in indices:
                indices[num] = indices[num] + [i]
            else:
                indices[num] = [i]

        pairs_count = 0
        for key in indices.keys():
            this_value_indices = indices[key]
            if len(this_value_indices) < 2:
                continue
            for i in range(len(this_value_indices)):
                for j in range(i + 1, len(this_value_indices)):
                    if (this_value_indices[i] * this_value_indices[j]) % k == 0:
                        pairs_count += 1

        return pairs_count


# https://leetcode.com/problems/find-first-palindromic-string-in-the-array/

# Discuss tab
# cleverly concise, though it relies on knowing Python iterator syntax
def firstPalindrome(self, words: List[str]) -> str:
    return next((word for word in words if word == word[::-1]), "")

# my solution
# time: worst O(n), avg O(nf) where f is frequency of palindromes in set
# space: O(1)
class Solution:
    def is_palindrome(self, word: str) -> bool:
        # Seems like this would be a bit faster than:  return word == word[::-1]
        # because it can "fail out" faster
        for i in range(len(word)):
            if word[i] != word[(len(word) - 1) - i]:
                return False
        return True

    def firstPalindrome(self, words: List[str]) -> str:
        for word in words:
            if self.is_palindrome(word):
                return word
        return ''


# https://leetcode.com/problems/replace-all-digits-with-characters/

# Discuss tab solution
# same solution as mine! wow!
# It was a little terser; they didn't bother with temp vars, but mine's arguably more readable

# my solution
# time: O(n)?; space: initially thought O(1) because drop constant? I think
# I'm wrong though, because we're creating a new data structure equal to input size, so O(n)
class Solution:
    def replaceDigits(self, s: str) -> str:
        output = list(s)
        for i in range(1, len(s), 2):
            to_shift = s[i - 1]
            shifted = chr(ord(to_shift) + int(s[i]))
            output[i] = shifted

        return ''.join(output)


# https://leetcode.com/problems/truncate-sentence/

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
