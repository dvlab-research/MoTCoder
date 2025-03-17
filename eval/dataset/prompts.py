ONE_SHOT_MOT_LABEL = """ Please perform the conversion of plain code into modular code. Here is an example.

## EXAMPLE CONVERSION
### INSTRUCTION
You are given three strings s1, s2, and s3. You have to perform the following operation on these three strings as many times as you want.
In one operation you can choose one of these three strings such that its length is at least 2 and delete the rightmost character of it.
Return the minimum number of operations you need to perform to make the three strings equal if there is a way to make them equal, otherwise, return -1.

Example 1:
Input: s1 = "abc", s2 = "abb", s3 = "ab"
Output: 2

Example 2:
Input: s1 = "dac", s2 = "bac", s3 = "cac"
Output: -1

### PLAIN CODE
```python
if __name__ == '__main__':
    s = input()
    if s == s[::-1]:
        print(s)
    else:
        start, end = 0, 0
        for i in range(len(s)):
            left, right = i, i
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            if right - left - 1 > end - start:
                start = left + 1
                end = right - 1
            left, right = i, i + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1            
            if right - left - 1 > end - start:
                start = left + 1
                end = right - 1
        print(s[start:end+1])
```
        
### MODULAR CODE
```python
def expandAroundCenter(s, left, right):
    ‘‘‘Expands around the center given by indices 'left' and 'right' while the characters at 'left' and 'right' are the same.’’’
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left, right

def update_indices_if_longer(left, right, start, end):
    ‘‘‘Updates start and end indices if the current palindrome is longer.’’’
    if right - left - 1 > end - start:
        start = left + 1
        end = right - 1
    return start, end

if __name__ == ‘__main__’:
    s = input()
    if s == s[::-1]:
        return s
    start, end = 0, 0
    for i in range(len(s)):
        left, right = expandAroundCenter(s, i, i)
        start, end = update_indices_if_longer(left, right, start, end)
        left, right = expandAroundCenter(s, i, i+1)
        start, end = update_indices_if_longer(left, right, start, end)
    return s[start:end+1]
```

## CONVERSION
### INSTRUCTION
{instruction}

### PLAIN CODE
```python
{code}
```

### MODULE-OF-THOUGHT CODE
"""

CORRECT_LABEL = """Please correct the following WRONG code.

### INSTRUCTION
{instruction}

### WRONG CODE
```python
{code}
```

### CORRECT CODE
"""

CLEAN_LABEL = """Optimize the code to improve readability.
- Optimize variable names to better reflect their purpose
- Add comments
- Follow the instructions and the meaning of the original code WITHOUT changing its functionality.

### INSTRUCTION
{instruction}

### ORIGINAL CODE
```python
{code}
```

### REWRITE CODE
"""

MOT_LABEL = """Optimize the code to improve readability.
- Optimize variable names to better reflect their purpose
- Add comments
- If there are code segments that are functionally clear and reusable, use functions.
- Follow the instructions and the meaning of the original code WITHOUT changing its functionality.

### INSTRUCTION
{instruction}

### WRONG CODE
```python
{code}
```

### CORRECT CODE
"""

FIX = """"""

PROMPT = {
    'test': """Develop a Python solution for the provided problem. Provide only the python code, without explanations. Please wrap your code answer using ```python ```.
### QUESTION:
{instruction}
### RESPONSE:""",
    'mot': MOT_LABEL,
    'correct_label': CORRECT_LABEL,
    'clean': CLEAN_LABEL,
    'query': '{instruction}',
    'fix': FIX
}