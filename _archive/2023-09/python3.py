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
# url: https://leetcode.com/problems/reverse-linked-list/

# (notes from LeetCode Solutions tab and/or ChatGPT)


# (my solution)
# time: O(n)
# space: O(n) because of call stack size?
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, current: Optional[ListNode], previous=None) -> Optional[ListNode]:
        if not current:
            return None
        next_node = current.next
        current.next = previous
        if not next_node:  # reached final node or is single node list
            return current
        else:
            return self.reverseList(next_node, current)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/longest-palindrome/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
#
# GPT-4's solution that makes things more concise while still retaining decent
# readability, though at the cost of needing an import and needing to understand
# the generator expression
from collections import Counter

class Solution:
    def longestPalindrome(self, s: str) -> int:
        counts = Counter(s)

        length = sum((v // 2) * 2 for v in counts.values())
        if length < len(s):  # there's at least one character with an odd count
            length += 1

        return length

# (my solution)
# time: O(n)
# space: O(n) worst in case of all unique characters, but across different cases,
# should be < O(n) on average
class Solution:
    def longestPalindrome(self, s: str) -> int:
        counts = {}
        for character in s:
            if character not in counts:
                counts[character] = 1
            else:
                counts[character] += 1

        length = 0
        odd_numbers = 0
        for value in counts.values():
            length += value
            if value % 2 != 0:
                odd_numbers += 1
        # subtract the non-even portions encountered, except leave one,
        # because a single character can be in the middle and it remains a
        # palindrome
        if odd_numbers:
            length -= odd_numbers - 1

        return length
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/climbing-stairs/

# (notes from LeetCode Solutions tab and/or ChatGPT)
#
# GPT-4's quite terse but readable solution
class Solution:
    def __init__(self):
        self.table = {0: 1, 1: 1}  # Base cases

    def climbStairs(self, n: int) -> int:
        if n in self.table:
            return self.table[n]

        self.table[n] = self.climbStairs(n - 1) + self.climbStairs(n - 2)
        return self.table[n]

# (my solution)
# time: O(n)?
# edit: that was correct!
# space: O(n)?
# edit: that was correct!
class Solution:
    def __init__(self):
        # cache Fibonacci values to avoid expensive compute during recursion
        self.table = {}

    def get_fibonacci(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        output = None
        if n not in self.table:
            output = self.get_fibonacci(n - 1) + self.get_fibonacci(n - 2)
        else:
            output = self.table[n]
        print(output)
        return output

    def climbStairs(self, n: int) -> int:
        # we notice a pattern where the answer aligns with the Fibonacci sequence
        # but offset by n+1
        cycles = n + 1
        for i in range(n):  # crucially, we go UP from the bottom small n values
            if i not in self.table:
                self.table[i] = self.get_fibonacci(i)
        return self.get_fibonacci(cycles)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/ransom-note/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
#
# cool solution from GPT-4 (likely pilfered from leetcode or whatever) that does
# space complexity of O(1); may or may not be faster, because it's creating
# strings every iteration
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        for letter in ransomNote:
            if letter in magazine:
                magazine = magazine.replace(letter, '', 1)
            else:
                return False
        return True

# (my solution)
# time: O(n)
# space: O(n) worst case like all unique letters; i have an intuition that
# improving this is possible
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        counts = {}
        for letter in magazine:
            if letter in counts:
                counts[letter] += 1
            else:
                counts[letter] = 1

        for letter in ransomNote:
            if letter not in counts or counts[letter] < 1:
                return False
            else:
                counts[letter] -= 1

        return True
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/first-bad-version/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# cleaner solution from GPT; not only is it more terse, but it is easier to
# reason about the exit condition by way of the left < right
class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n  # Initialize left to 1 as version numbers start from 1
        while left < right:
            mid = (left + right) // 2  # Use integer division to find the midpoint
            if isBadVersion(mid):
                right = mid  # If mid is bad, search the left half
            else:
                left = mid + 1  # If mid is good, search the right half
        return left  # When left and right converge, return the result

# (my solution)
# time: O(log n)
# space: O(1)
#
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:
import math

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left = 0  # 0 is needed to be able to land on a midpoint of 1
        right = n
        current_midpoint = math.ceil(n / 2)
        while True:
            if isBadVersion(current_midpoint):
                # search left half
                right = current_midpoint
                new_midpoint = left + ceil((right - left) / 2)
                if new_midpoint == current_midpoint:
                    return current_midpoint
            else:
                # search right half
                left = current_midpoint
                new_midpoint = current_midpoint + ceil((right - left) / 2)
                if new_midpoint == current_midpoint:
                    return current_midpoint
            current_midpoint = new_midpoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/implement-queue-using-stacks/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# I didn't see anything too impressive in the Solutions tab on LeetCode
#
# GPT-4's DRYer version of my code using a helper method:
class MyQueue:
    def __init__(self):
        self.list1 = []  # for push()
        self.list2 = []  # for peek(), pop()

    def push(self, x: int) -> None:
        self.list1.append(x)

    def _prepareList2(self):
        if not self.list2:
            while self.list1:
                self.list2.append(self.list1.pop())

    def pop(self) -> int:
        self._prepareList2()
        return self.list2.pop()

    def peek(self) -> int:
        self._prepareList2()
        return self.list2[-1]

    def empty(self) -> bool:
        return not self.list1 and not self.list2

# (my solution)
# time: O(1) best, O(n) worst?
#
# EDIT: amortized O(1)
# space: O(n)
class MyQueue:
    def __init__(self):
        self.list1 = []  # for push()
        self.list2 = []  # for peek(), pop()

    def push(self, x: int) -> None:
        self.list1.append(x)

    def pop(self) -> int:
        if not self.list2:
            while self.list1:
                self.list2.append(self.list1.pop())
        return self.list2.pop()

    def peek(self) -> int:
        if not self.list2:
            while self.list1:
                self.list2.append(self.list1.pop())
        return self.list2[-1]

    def empty(self) -> bool:
        return len(self.list2) == 0 and len(self.list1) == 0
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/linked-list-cycle/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
#
# ideal solution with space O(1) using Tortoise and the Hare (or Floyd's Cycle
# Detection Algorithm)
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False

        tortoise, hare = head, head.next

        while hare != tortoise:
            if not hare or not hare.next:
                return False
            tortoise = tortoise.next
            hare = hare.next.next

        return True

# GPT-4 improvement to my code: we don't have to check the length of the set; we
# can simply see if the current node is already in the set: `if head in seen`
#
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        seen = set()
        while head:
            if head in seen:  # had seen node earlier; hence, cycle
                return True
            seen.add(head)
            head = head.next
        return False

# (my solution)
# time: worst is O(n) for no-cycles linked list
# space: I think O(1)? we're just storing references?
#
# EDIT: I was wrong: it's O(n) for a no-cycles linked list; even though we're
# only storing references (pointers), they're still growing in proportion to the
# size of the input and still take up memory
#
# Notes from GPT:
# We're interested in growth. It's about how memory usage grows as the input size increases. Even if a reference is much smaller than a full object, if you need a linear number of references relative to the size of the input, the space complexity is O(n).
# Constants don't matter in Big O notation. If a full object took up, let's say, 100 units of memory and a reference took up 1 unit, in Big O notation, we'd still disregard that factor of 100 and focus on how the number of these entities grows with the size of the input.
# It standardizes discussion. Whether we're storing full objects or references, if we need a linear number of them, we say it's O(n). This standardization makes it easier to communicate and reason about algorithmic efficiency without getting bogged down in implementation specifics.
# In practice, if you're trying to optimize real-world memory usage, then the absolute sizes (like the difference between the size of an object and a reference) can become important. But for algorithmic analysis, we abstract away those details to focus on growth patterns.
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        seen = set()
        while head.next:
            old_length = len(seen)
            seen.add(head)
            if len(seen) == old_length:  # had seen node earlier; hence, cycle
                return True
            head = head.next
        return False
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/balanced-binary-tree/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# a solution from LeetCode; it's short and clever, however, I feel it sacrifices
# readability: it's using an integer as both the height and whether a subtree is
# balanced, so it requires special knowledge that -1 means an unbalanced subtree
class Solution(object):
    def isBalanced(self, root):
        return (self.Height(root) >= 0)
    def Height(self, root):
        if root is None:  return 0
        leftheight, rightheight = self.Height(root.left), self.Height(root.right)
        if leftheight < 0 or rightheight < 0 or abs(leftheight - rightheight) > 1:  return -1
        return max(leftheight, rightheight) + 1

# (my solution)
# it's a mess, but it does work
#
# time: O(n)
# space: O(1)
# EDIT: I was wrong; it's O(n) for a skewed tree, O(log n) AKA O(h) for a
# balanced tree
#
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def get_height(self, root, stats=(0, True)):
        height = stats[0]
        is_balanced = stats[1]
        # NOTE: you can instead do a, b = tuple (tuple unpacking)
        if root is None:  # no children; cannot be unbalanced from here on out
            return (0, True)
        left_height = 0
        right_height = 0
        left_balanced = True
        right_balanced = True
        if root.left:
            left_height, left_balanced = self.get_height(root.left, (height, is_balanced))
        if root.right:
            right_height, right_balanced = self.get_height(root.right, (height, is_balanced))
        # left or right trees unbalanced OR heights of left and right are unbalanced
        if not left_balanced or not right_balanced or abs(left_height - right_height) > 1:
            return (0, False)
        # add 1 to account for parent node height contribution
        return (max(left_height, right_height) + 1, True)

    def isBalanced(self, root):
        # empty tree case and single node case
        if not root or (not root.left and not root.right):
            return True
        left_height, left_balanced = self.get_height(root.left)
        right_height, right_balanced = self.get_height(root.right)
        return left_balanced and right_balanced and abs(left_height - right_height) <= 1
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# GPT-4's solution, which is both shorter and more elegant: "if p and q are
# smaller than the current node value, search left (which is guaranteed to be
# smaller vals) and if p and q are larger, search right/larger section of tree"
#
# the time complexity here is better, because it achieves O(h) -- which is
# O(log n) in a balanced tree, but as bad as O(n) on a skewed tree (e.g. a
# straight line)
# It's able to do this by skipping half the tree: going to the smaller/larger
# child
# My code below was always doing all branches
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root:
            return None
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

# (my solution; second, successful attempt after getting a couple hints)
#
# the key to the puzzle is that the LCA in a BST will always been greater than
# or equal to the smaller of the two node values, and smaller than or equal to
# the larger of the two node values
#
# my solution still isn't great, though. it isn't taking advantage of the BST
# properties. with GPT's solution above, on a balanced tree, it's able to
# eliminate half of the remaining tree at each step
#
# time: O(log n)? I think the properties of the search tree will mean it should
# avoid O(n)
# EDIT: I was wrong. Time complexity is O(n) in the worst case for THIS code.
# but my intuition was right in that in better code (above), it is indeed
# O(log n) AKA O(h)
#
# space: O(1) because only references to nodes, not creating anything
# Definition for a binary tree node.
# EDIT: I was wrong. Space complexity is O(h) where h is the height of the tree
# to account for recursion stack depth
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root or root.val is None:  # val of root could be 0 and thus falsey, so check for null instead
            return
        if root.val >= min(p.val, q.val) and root.val <= max(p.val, q.val):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is not None:
            return left
        if right is not None:
            return right

# (my solution; first, failing attempt)
# time:
# space:
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.sequences = []
        self.nodes = {}

    def lowestCommonAncestorCrawler(self, root, p, q, accumulator):
        if root and root.val:
            accumulator += [root.val]
            self.nodes[root.val] = root
        else:
            return
        self.sequences.append(accumulator)
        self.lowestCommonAncestorCrawler(root.left, p, q, accumulator)
        self.lowestCommonAncestorCrawler(root.right, p, q, accumulator)

    def lowestCommonAncestor(self, root, p, q):
        self.lowestCommonAncestorCrawler(root, p, q, [])
        print(self.sequences)
        print(self.nodes)
        for sequence in self.sequences:
            if p.val in sequence:
                p_index = sequence.index(p.val)
            else:
                continue
            if q.val in sequence:
                q_index = sequence.index(q.val)
            else:
                continue
            smallest_index = min(p_index, q_index)
            if smallest_index > 0:
                return self.nodes[sequence[smallest_index - 1]]
            if abs(p_index - q_index) == 1:
                value = sequence[smallest_index]
                return p if p.val == value else q

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/flood-fill/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
#
# I was wrong about the time complexity. It's basically O(n), although it's
# more precisely described as O(m * n) where m and n are the dimensions of the
# pixel matrix:
# "In computational geometry and algorithms dealing with matrices or grids,
# O(m * n) is often preferred for clarity."
#
# here's GPT-4's streamlined version of my solution, which cleans up a lot of
# the verbosity in mine by eliminating extraneous temp vars and doing fewer
# checks in loops
from typing import List

class Solution:
    def get_neighbors(self, x, y, width, height):
        neighbors = []
        if x - 1 >= 0:
            neighbors.append((y, x - 1))
        if x + 1 < width:
            neighbors.append((y, x + 1))
        if y - 1 >= 0:
            neighbors.append((y - 1, x))
        if y + 1 < height:
            neighbors.append((y + 1, x))
        return neighbors

    def floodFill(self, image: List[List[int]], sr: int, sc: int, target_color: int) -> List[List[int]]:
        height, width = len(image), len(image[0])
        start_color = image[sr][sc]
        if start_color == target_color:
            return image

        needs_painting = {(sr, sc)}

        while needs_painting:
            y, x = needs_painting.pop()
            image[y][x] = target_color

            for new_y, new_x in self.get_neighbors(x, y, width, height):
                if image[new_y][new_x] == start_color:
                    needs_painting.add((new_y, new_x))

        return image
# and here's a recursive solution using depth-first search; time and space
# complexity remain the same, since there's a call stack involved in the
# recursion
from typing import List

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, target_color: int) -> List[List[int]]:
        start_color = image[sr][sc]
        if start_color == target_color:
            return image

        def dfs(x, y):
            if x < 0 or x >= len(image[0]) or y < 0 or y >= len(image):
                return
            if image[y][x] != start_color:
                return

            image[y][x] = target_color

            dfs(x+1, y)
            dfs(x-1, y)
            dfs(x, y+1)
            dfs(x, y-1)

        dfs(sc, sr)
        return image

# (my solution)
# time: I think it might be O(n log n)? there is a bit of overlap when
# get_neighbors is performed [edit: was wrong about that]
# corrected version: O(m * n)
# space: O(n) worst case when every pixel needs to be painted
# corrected version: O(m * n)
class Solution:
    # return up to four valid neighbors within contraints of image size and paint color
    def get_neighbors(self, image, start_color, target_color, width, height, x, y):
        neighbors = []
        if x - 1 >= 0 and image[y][x - 1] == start_color and image[y][x - 1] != target_color:
            neighbors.append((y, x - 1))
        if x + 1 < width and image[y][x + 1] == start_color and image[y][x + 1] != target_color:
            neighbors.append((y, x + 1))
        if y - 1 >= 0 and image[y - 1][x] == start_color and image[y - 1][x] != target_color:
            neighbors.append((y - 1, x))
        if y + 1 < height and image[y + 1][x] == start_color and image[y + 1][x] != target_color:
            neighbors.append((y + 1, x))
        return neighbors

    def floodFill(self, image: List[List[int]], sr: int, sc: int, target_color: int) -> List[List[int]]:
        needs_painting = set()  # for all pixels that need to change
        candidates = [(sr, sc)]
        height = len(image)
        width = len(image[0])
        START_COLOR = image[sr][sc]

        while candidates:
            # check to see if current pixels under evaluation need to be painted
            new_should_paint = set()
            for candidate in candidates:
                actual_color = image[candidate[0]][candidate[1]]
                if actual_color == START_COLOR and actual_color != target_color:
                    new_should_paint.add(candidate)

            if not new_should_paint:  # if no valid new pixels, stop
                break
            needs_painting.update(new_should_paint)

            # find valid neighbors of all current pixels under evaluation
            neighbors = set()
            for candidate in candidates:
                new_neighbors = self.get_neighbors(
                    image,
                    START_COLOR,
                    target_color,
                    width,
                    height,
                    candidate[1],
                    candidate[0]
                )
                neighbors.update(new_neighbors)

            # create a new list of candidate pixels for later evaluation when
            # the loop repeats
            candidates = []
            for neighbor in neighbors:
                if neighbor not in needs_painting:  # skip pixels we know about
                    candidates.append(neighbor)

        # actually paint the pixels
        for item in needs_painting:
            image[item[0]][item[1]] = target_color
        return image
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/binary-search/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# this solution is able to remove my extraneous guards/catches and simplifies
# the code; it calculates the midpoint directly instead of difference between
# upper and lower plus lower; and it moves the pointer around more effectively
# by setting upperbound to below pointer if candidate is too high and vice versa
# for candidate too lower, set lowerbound to just above pointer
#
# time and space are still the same, but this has fewer special cases
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lowerbound = 0
        upperbound = len(nums) - 1
        while lowerbound <= upperbound:
            pointer = (upperbound + lowerbound) // 2
            candidate = nums[pointer]
            if candidate == target:
                return pointer
            elif candidate > target:
                upperbound = pointer - 1
            else:
                lowerbound = pointer + 1
        return -1

# (my solution)
# time: O(log n)
# space: O(1)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lowerbound = 0
        upperbound = len(nums)
        while True:
            pointer = (upperbound - lowerbound) // 2 + lowerbound
            if pointer >= len(nums) or pointer < 0:
                return -1
            candidate = nums[pointer]
            if candidate == target:
                return pointer
            if pointer in [lowerbound, upperbound]:
                return -1
            elif candidate > target:
                upperbound = pointer
            else:
                lowerbound = pointer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/valid-anagram/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# here's one from Solutions tab that has O(1) space complexity assuming we're
# only dealing with lowercase English letters, which in this case we are
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        count = [0] * 26

        # Count the frequency of characters in string s
        for x in s:
            count[ord(x) - ord('a')] += 1

        # Decrement the frequency of characters in string t
        for x in t:
            count[ord(x) - ord('a')] -= 1

        # Check if any character has non-zero frequency
        for val in count:
            if val != 0:
                return False

        return True

# here's an option that uses an imported library:
from collections import Counter

def isAnagram(self, s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

# here's a non-import solution that uses only a single dictionary and some slick
# ideas like a default value for non-existent keys:
def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    counts = {}

    for char in s:
        counts[char] = counts.get(char, 0) + 1 # get returns 0 if not exists

    for char in t:
        if char not in counts:
            return False
        counts[char] -= 1
        if counts[char] < 0:
            return False

    return True

# (my solution)
# time: O(n)
# space: O(n) worst case when each char is unique, e.g. "abcdefg"
class Solution:
    def map_counts(self, string):
        counts_map = {}
        for char in string:
            if char not in counts_map:
                counts_map[char] = 1
            else:
                counts_map[char] += 1
        return counts_map

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        s_counts = self.map_counts(s)
        t_counts = self.map_counts(t)

        for key, value in s_counts.items():
            if key not in t_counts or value != t_counts[key]:
                return False
        return True
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/invert-binary-tree/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# I was wrong about the space complexity:
#
# "the space complexity is not O(1). The space complexity is actually O(h),
# where h is the height of the tree, due to the recursive call stack. In the
# worst case, this could be O(n) for a skewed tree and O(log n) for a balanced
# tree."
#
# "Space complexity is not just about the new variables you explicitly allocate;
# it also includes the memory taken up by the function call stack, especially
# for recursive functions. Each recursive call adds a new frame to the call
# stack, and these frames take up memory. So when you're analyzing space
# complexity, you need to consider both the variables you allocate and the call
# stack."

# (my solution)
# time: O(n)
# space: O(1) [this is wrong; see above]
# Proud of myself: I got this one without any hints, just using hazy memory of
# when I'd done similar puzzles in the past!
#
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# url: https://leetcode.com/problems/valid-palindrome/description/

# (notes from LeetCode Solutions tab and/or ChatGPT)
# here's a way to do O(n/2) for the reverse-checking, though it's considered
# O(n) as constant factors are usually ommited
#
# but it was interesting to find, and I went down a rabbithole of refreshing my
# memory on two's complement / bitwise operators
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = [c.lower() for c in s if c.isalnum()]
        return all (s[i] == s[~i] for i in range(len(s)//2))

# (my solution)
# time: O(n)
# space: O(n)
class Solution:
    def isPalindrome(self, s: str) -> bool:
        lowercase = s.lower()
        stripped = ''.join(filter(str.isalnum, lowercase))
        return stripped == ''.join(reversed(stripped))
# ---------------------------------------------------------------------------

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