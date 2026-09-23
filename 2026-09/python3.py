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
# url: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

# (solution and notes from LeetCode Solutions tab and/or AI model)


# (my solution)
# time:
# space:
#
# IN PROGRESS!
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        pairs = []

        for i, price in enumerate(prices):
            if i == 0:
                continue

            temp_pairs = []
            temp_pairs.append((prices[i-1], i-1))
            temp_pairs.append((prices[i], i))

            if temp_pairs[0][0] < temp_pairs[1][0] and temp_pairs[0][1] < temp_pairs[1][1]:
                if not pairs:
                    pairs.append(temp_pairs[0])
                    pairs.append(temp_pairs[1])
                else:
                    if temp_pairs[0][0] < pairs[0][0] and temp_pairs[0][1] < pairs[1][1]:
                        pairs[0] = temp_pairs[0]
                    if temp_pairs[1][0] > pairs[1][0] and temp_pairs[1][1] > pairs[0][1]:
                        pairs[1] = temp_pairs[1]

        if not pairs:
            return 0
        else:
            return pairs[1][0] - pairs[0][0]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/merge-two-sorted-lists/description/

# (solution and notes from LeetCode Solutions tab and/or AI model)


# (my solution)
# time: O(n + m) -- the two lists
# space: O(1)  -- because we're just reconnecting things, not creating new nodes
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: ListNode | None, list2: ListNode | None) -> ListNode | None:
        if not list1:
            return list2
        elif not list2:
            return list1

        merged_chain = ListNode()
        head = merged_chain

        while list1 or list2:
            node_to_add = None
            temp = None


            # otherwise build chain
            print("list1.val", list1.val)
            print("list2.val", list2.val)
            if list2.val < list1.val:
                temp = list2.next
                node_to_add = list2
                node_to_add.next = None
                merged_chain.next = node_to_add
                list2 = temp
                merged_chain = merged_chain.next
            else:
                temp = list1.next
                node_to_add = list1
                node_to_add.next = None
                merged_chain.next = node_to_add
                list1 = temp
                merged_chain = merged_chain.next

            # when one of the lists runs out
            if not list1:
                merged_chain.next = list2
                return head.next
            if not list2:
                merged_chain.next = list1
                return head.next
        return head.next
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/valid-parentheses/

# (solution and notes from LeetCode Solutions tab and/or AI model)
#
# after a couple small hints, like using a list as a stack, and realizing
# slicing off the last two entries in the list copies the whole list thus
# risking O(n^2) again (doing an O(n) for every element in the string)
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) < 2 or s[0] in [']', '}', ')']:
            return False

        parens = []
        parensMap = { ')': '(', ']': '[', '}': '{'}
        for char in s:
            parens.append(char)
            if len(parens) > 1 and char in [']', '}', ')']:
                if parens[-2] == parensMap[char]:  # if two back is opening paren
                    parens.pop()  # remove the closed pair
                    parens.pop()  # remove the closed pair
        return not parens  # every pair closed and remove

# (my solution)
# time: O(n^2) because string appends can add up
# space: O(n)
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) < 2 or s[0] in [']', '}', ')']:
            return False

        parens = ""
        parensMap = { ')': '(', ']': '[', '}': '{'}
        for char in s:
            parens += char
            if len(parens) > 1 and char in [']', '}', ')']:
                if parens[-2] == parensMap[char]:  # if two back is opening paren
                    parens = parens[:-2]  # remove the closed pair
        return not parens  # every pair closed and removed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/two-sum/description/

# (solution and notes from LeetCode Solutions tab and/or AI model)
#
# I half-remembered the solution but I needed a hint from the LLM to get the
# hint about the seen dictionary
#
# My new version which is still O(n), since constant factors (2n) are dropped:
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}
        for i, number in enumerate(nums):
            if number in seen:
                seen[number] = seen[number] + [i]
            else:
                seen[number] = [i]
        for key in seen.keys():
            difference = target - key
            if difference in seen:
                if key == difference:
                    if len(seen[difference]) > 1:
                        return seen[difference]
                else:
                    return [seen[key][0], seen[difference][0]]
# nicer version from LLM that avoids the two passes:
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}

        for i, number in enumerate(nums):
            difference = target - number

            if difference in seen:
                return [seen[difference], i]

            seen[number] = i

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
