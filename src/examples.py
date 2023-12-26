__all__ = [
    'example_question', 
    'example_code',
    'example_code_codellama',
    'example_modules',
    'example_modules_codellama',
    'example_modular_code',
    'example_modular_code_codellama',
    'example_question_leetcode',
    'example_code_leetcode'
]

example_question = """
Polycarp has $n$ different binary words. A word called binary if it contains only characters '0' and '1'. For example, these words are binary: "0001", "11", "0" and "0011100".

Polycarp wants to offer his set of $n$ binary words to play a game "words". In this game, players name words and each next word (starting from the second) must start with the last character of the previous word.

The first word can be any. For example, these sequence of words can be named during the game: "0101", "1", "10", "00", "00001".

Word reversal is the operation of reversing the order of the characters. For example, the word "0111" after the reversal becomes "1110", the word "11010" after the reversal becomes "01011".

Probably, Polycarp has such a set of words that there is no way to put them in the order correspondent to the game rules. In this situation, he wants to reverse some words from his set so that: the final set of $n$ words still contains different words (i.e. all words are unique); there is a way to put all words of the final set of words in the order so that the final sequence of $n$ words is consistent with the game rules.

Polycarp wants to reverse minimal number of words. Please, help him.

-----Input----
The first line of the input contains one integer $t$ ($1 \le t \le 10^4$) -- the number of test cases in the input. Then $t$ test cases follow.

The first line of a test case contains one integer $n$ ($1 \le n \le 2\cdot10^5$) -- the number of words in the Polycarp's set. Next $n$ lines contain these words. All of $n$ words aren't empty and contains only characters '0' and '1'. The sum of word lengths doesn't exceed $4\cdot10^6$. All words are different.

Guaranteed, that the sum of $n$ for all test cases in the input doesn't exceed $2\cdot10^5$. Also, guaranteed that the sum of word lengths for all test cases in the input doesn't exceed $4\cdot10^6$.

-----Output----
Print answer for all of $t$ test cases in the order they appear.

If there is no answer for the test case, print -1. Otherwise, the first line of the output should contain $k$ ($0 \le k \le n$) -- the minimal number of words in the set which should be reversed. The second line of the output should contain $k$ distinct integers -- the indexes of the words in the set which should be reversed. Words are numerated from $1$ to $n$ in the order they appear. If $k=0$ you can skip this line (or you can print an empty line). If there are many answers you can print any of them.

-----Example-----
Input
4
4
0001
1000
0011
0111
3
010
101
0
2
00000
00001
4
01
001
0001
00001
Output
1
3
-1
0
2
1 2
"""

example_code_main = """
for _ in range(int(input())):
    n = int(input())
    mass = []
    zo = 0
    oz = 0
    zz = 0
    oo = 0
    ozs = []
    zos = []
    ozss = set()
    zoss = set()
    for j in range(n):
        k = input()
        mass.append(k)
        if k[0] == '0' and k[-1] == '1':
            zoss.add(k)
            zos.append(j + 1)
            zo += 1
        elif k[0] == '1' and k[-1] == '0':
            ozss.add(k)
            ozs.append(j + 1)
            oz += 1
        elif k[0] == '0' and k[-1] == '0':
            zz += 1
        else:
            oo += 1
    if zz and oo and not oz and not zo:
        print(-1)
        continue
    else:
        if zo > oz:
            print((zo - oz) // 2)
            ans = []
            need = (zo - oz) // 2
            i = 0
            while need:
                zzz = mass[zos[i] - 1][len(mass[zos[i] - 1]) - 1:: -1]
                if zzz not in ozss:
                    ans.append(zos[i])
                    need -= 1
                i += 1
            print(*ans)
        else:
            print((oz - zo) // 2)
            ans = []
            need = (oz - zo) // 2
            i = 0
            while need:
                zzz = mass[ozs[i] - 1][len(mass[ozs[i] - 1]) - 1:: -1]
                if zzz not in zoss:
                    ans.append(ozs[i])
                    need -= 1
                i += 1
            print(*ans)
"""

example_code = f"""```python
{example_code_main}
```
"""

example_code_codellama = f"""```[PYTHON]
{example_code_main}
[/PYTHON]
"""

example_modular_code_main = """
import collections

def count_start_end_chars(words):
    start_count = collections.defaultdict(int)
    end_count = collections.defaultdict(int)
    for word in words:
    start_count[word[0]] += 1
    end_count[word[-1]] += 1
    return start_count, end_count

def reverse_words(words, reversed_indices):
    reversed_words = []
    for i, word in enumerate(words):
    if i in reversed_indices:
    reversed_words.append(word[::-1])
    else:
    reversed_words.append(word)
    return reversed_words

def solve_task(words):
    start_count, end_count = count_start_end_chars(words)
    characters_with_difference = []
    for char in start_count:
    if abs(start_count[char] - end_count[char]) > 1:
    characters_with_difference.append(char)
    reversed_indices = []
    for char in characters_with_difference:
    difference = abs(start_count[char] - end_count[char])
    reverse_count = difference // 2
    if start_count[char] < end_count[char]:
    indices = [i for i, word in enumerate(words) if word.startswith(char)]
    reversed_indices.extend(indices[:reverse_count])
    else:
    indices = [i for i, word in enumerate(words) if word.endswith(char)]
    reversed_indices.extend(indices[:reverse_count])
    reversed_words = reverse_words(words, reversed_indices)
    total_reversed = len(reversed_indices)
    return total_reversed, reversed_words

t = int(input())
for _ in range(t):
    n = int(input())
    words = []
    for _ in range(n):
    words.append(input())
    total_reversed, reversed_words = solve_task(words)
    print(total_reversed)
    if total_reversed != 0:
    print(*reversed_words)
"""


example_modular_code = f"""
```python
{example_modular_code_main}
```
"""

example_modular_code_codellama = f"""
[PYTHON]
{example_modular_code_main}
[/PYTHON]
"""

example_module_1 = """
def count_start_end_chars(words):
    \"\"\"
    Description: This function counts the number of words that start and end with each character.
    Input:
    words (list): A list of binary words.
    Output:
    start_count (defaultdict): A dictionary containing the count of words that start with each character.
    end_count (defaultdict): A dictionary containing the count of words that end with each character.
    \"\"\"
"""

example_module_2 = """
def reverse_words(words, reversed_indices):
    \"\"\"
    Description: This function reverses the specified words in the given list.
    Input:
    words (list): A list of binary words.
    reversed_indices (list): A list of indices indicating the words to be reversed.
    Output:
    reversed_words (list): A new list of words with the specified words reversed.
    \"\"\"
"""

example_modules = f"""
```module
{example_module_1}
```

```module
{example_module_2}
```"""

example_modules_codellama = f"""
[MODULE]
{example_module_1}
[/MODULE]

[MODULE]
{example_module_2}
[/MODULE]
"""

example_question_leetcode = f"""
```python
def twoSum(nums, target):
    \"\"\"
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    You can return the answer in any order.

    Example 1:

    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
    Example 2:

    Input: nums = [3,2,4], target = 6
    Output: [1,2]
    Example 3:

    Input: nums = [3,3], target = 6
    Output: [0,1]
    \"\"\"
```
"""

example_code_leetcode = """
```python
def twoSum(nums, target):
    l = len(nums)
    for i in range(l - 1):
        for j in range(i + 1, l):
            if nums[i] + nums[j] == target:
                return [i, j]
```
"""