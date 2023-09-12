# vvvv template vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ---------------------------------------------------------------------------
# url:

# (notes from LeetCode Solutions tab and/or ChatGPT)


# (my solution)
# time:
# space:


# ---------------------------------------------------------------------------
# ^^^^ template ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/submissions/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# GPT-4 suggested a clever way of simplifying the code while still maintaining
# readability:
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest = prices[0]
        max_profit = 0

        for item in prices[1:]:
            lowest = min(lowest, item)
            max_profit = max(max_profit, item - lowest)

        return max_profit

# (my solution)
# time: O(n)
# space: O(1)
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest = prices[0]
        max_profit = 0

        for item in prices[1:]:
            if item < lowest:
                lowest = item
            elif item > lowest:
                new_max_profit = item - lowest
                if new_max_profit > max_profit:
                    max_profit = new_max_profit

        return max_profit
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/merge-two-sorted-lists/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# this solution is functionally equivalent, but it greatly simplifies the code
#
# it uses the original lists, and it uses `current` as a pointer that can
# bounce back and forth between the lists, taking the smaller (or equal) value
#
# to have something clean to return, it starts with a dummy empty ListNode, to
# which we append new nodes, and cleverly return the dummy's .next instead of
# dummy itself, essentially moving "forward" one step away from the dummy
#
# desipte us creating a new ListNode(), there's effectively zero extra space
# being used and space complexity is still O(1); this is because dummy/curent
# are merely referring to existing nodes, not creating any more additional nodes
# (aside from the trivial initial dummy node)
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy

        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next

            current = current.next

        current.next = list1 or list2

        return dummy.next

# (my solution)
# time: O(n)
# space: O(1) because in-place modification
# Definition for singly-linked list
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # deal with edge of case of either or both lists being None/null:
        if not list1:
            return list2
        elif not list2:
            return list1

        # deal with second list starting val > first list starting value:
        if list2.val >= list1.val:
            first = list1
            second = list2
            final_list = list1
        else:
            first = list2
            second = list1
            final_list = list2

        while first and second:
            # second val is still too large to fit before next node, so move
            # forward in first list
            if first.next and second.val > first.next.val:
                first = first.next
                continue

            # point new node's next to next node in original list
            new_node = ListNode(second.val, first.next)
            # update original list next to be this new node
            first.next = new_node
            # move forward in both lists
            first = new_node
            second = second.next

        return final_list
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/valid-parentheses/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# GPT-4 suggested this slick solution where we have a map of closer to opener,
# which simplifies the conditional
# it also combines the conditionals for a one-liner conditional
# and the in <string> method instead of my ["l", "i", "s", "t"] is clever
class Solution:
    def isValid(self, s: str) -> bool:
        open_brackets = []
        bracket_map = {')': '(', '}': '{', ']': '['}

        for bracket in s:
            if bracket in "({[":
                open_brackets.append(bracket)
            else:
                if not open_brackets or open_brackets[-1] != bracket_map[bracket]:
                    return False
                open_brackets.pop()

        return len(open_brackets) == 0

# (my solution)
# time: O(n)
# space: O(n) -- worst is all open brackets
class Solution:
    def isValid(self, s: str) -> bool:
        open_brackets = []
        for bracket in s:
            if bracket in ["(", "[", "{"]:
                open_brackets.append(bracket)
            else:
                if not open_brackets: # close bracket with no opens available
                    return False
                if (bracket == ")" and open_brackets[-1] == "(") or \
                (bracket == "]" and open_brackets[-1] == "[") or \
                (bracket == "}" and open_brackets[-1] == "{"):
                    open_brackets.pop()
                else:
                    return False
        return len(open_brackets) == 0
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/two-sum/

# notes from LeetCode Solutions tab and/or ChatGPT
# with GPT-4's hints, made code cleaner and now we don't loop through things
# twice
# time and space are still O(n) in worst case
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        table = {}
        for i, num in enumerate(nums):
            if table.get(num) == None:
                table[num] = i
            difference = target - num
            if table.get(difference) != None:
                if table[difference] != i:
                    return [i,table[difference]]
# and here's a super clean solution from leetcode solutions tab:
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numMap = {}
        n = len(nums)

        for i in range(n):
            complement = target - nums[i]
            if complement in numMap:
                return [numMap[complement], i]
            numMap[nums[i]] = i

        return []  # No solution found

# my solution
# time: O(n)
# space: O(n) (worst-case scenario, every number is unique, requiring n slots)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        table = {}
        for i, num in enumerate(nums):
            if not table.get(num):
                table[num] = [i]
            else:
                table[num].append(i)

        for key in table.keys():
            difference = target - key
            if table.get(difference):
                if len(table[difference]) == 1:
                    if table[difference][0] != table[key][0]:
                        return [table[key][0], table[difference][0]]
                else:
                    return [table[difference][0], table[difference][1]]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/can-place-flowers/?envType=study-plan-v2&envId=leetcode-75

# notes from LeetCode Solutions tab and/or ChatGPT
# solution from mulliganaceous on leetcode that's quite terse:
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        count = 1
        beds = 0
        for bed in flowerbed:
            if bed:
                beds += (count - 1) // 2
                count = 0
            else:
                count += 1

        beds += count // 2

        return beds >= n

# my solution
# time: O(n)
# space: O(1)
class Solution:
    def getCapacity(self, capacity, counter):
        if counter == 2:
            return capacity + 1
        else:
            return capacity + ceil(counter / 2)

    def canPlaceFlowers(self, flowerBed: List[int], n: int) -> bool:
        capacity = 0
        counter = 0
        flowersSeen = 0

        # special case of [0]
        if len(flowerBed) == 1 and flowerBed[0] == 0:
            return True

        for i, plot in enumerate(flowerBed):
            if not plot: # empty space
                counter += 1
            else:
                flowersSeen += 1
                if flowersSeen == 1:
                    if counter != 1:
                        capacity = self.getCapacity(capacity, counter)
                elif counter != 0:
                    capacity += ceil(counter / 2 - 1)
                counter = 0

        # end zeroes
        if counter > 1:
            capacity = self.getCapacity(capacity, counter)

        return capacity >= n
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/?envType=study-plan-v2&envId=leetcode-75

# notes from LeetCode Solutions tab and/or ChatGPT
# someone on the site noticed you can get O(1) space complexity by modifying the list
# after doing the comparison, e.g.:
for idx, candy in enumerate(candies):
    canHaveMaxCandies = (candy + extraCandies) >= maxNumCandies
    candies[idx] = canHaveMaxCandies

# my solution
# time: O(n)
# space: O(n)
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        largest = max(candies)
        has_greatest = []
        for item in candies:
            has_greatest.append(item + extraCandies >= largest)
        return has_greatest
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/greatest-common-divisor-of-strings/?envType=study-plan-v2&envId=leetcode-75

# discuss tab solution
# this is actually GPT-4's solution; quite clever:
# time -- I was incorrect:
# O(n*m) because we loop through both strings once; "The string
# multiplication and comparison take time proportional to the lengths of both strings."
#
# space -- I was correct; GPT-4 wants to use this notation:, O(min(n, m))
# more commonly you'd probably just have O(n) and note that n is the shorter string
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        for i in range(min(len(str1), len(str2)), 0, -1):
            chunk = str1[:i]
            if str1[:i] == str2[:i]:
                if str1.replace(chunk, "") == "" and str2.replace(chunk, "") == "":
                    return chunk
        return ""

# my solution
# I had to get a hint; went down a wrong rabbithole with .count()
# time: O(n)?
# space: O(m) where m is the shorter of the two strings??
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        size = 0
        candidate = ""
        while True:
            size += 1
            if size > len(str2) or size > len(str1):
                return candidate
            chunk = str1[0:size]
            if chunk * (len(str1) // size) == str1 and \
            chunk * (len(str2) // size) == str2:
                candidate = chunk
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/merge-strings-alternately/?envType=study-plan-v2&envId=leetcode-75

# discuss tab solution
# near as I can tell, there's no way to improve the time or space complexity of
# O(n)
#
# there was this "clever" one-liner in the Discussion tab (below); it's actually
# not that bad in terms of understandability given it's so terse; a key to
# understanding it is: The zip function stops pairing when the shortest input
# iterable is exhausted.
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return "".join(a + b for a, b in zip(word1, word2)) + word1[len(word2):] + word2[len(word1):]

# my solution
# I don't believe there is a way to get below O(n) with either time or space
# I thought about doing a dual pointers thing, but the immutability of strings
# in Python makes it difficult to do any kind of in-place swapping
# time: O(n)
# space: O(n)
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        shortest_length = len(word1) if len(word1) <= len(word2) else len(word2)
        output = []
        for i in range(shortest_length):
            output += word1[i]
            output += word2[i]

        if (len(word1) != len(word2)):
            longest_word = word1 if len(word1) > len(word2) else word2
            output += longest_word[shortest_length:]
        return "".join(output)
# ---------------------------------------------------------------------------