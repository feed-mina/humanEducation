-- V28: leetcode_problems 테이블 생성 + Top Interview Questions 시드 데이터 (57문제)

CREATE TABLE leetcode_problems (
    id            SERIAL PRIMARY KEY,
    title         VARCHAR(200) NOT NULL,
    slug          VARCHAR(200) NOT NULL UNIQUE,
    difficulty    VARCHAR(10)  NOT NULL,  -- Easy / Medium / Hard
    category      VARCHAR(50)  NOT NULL,  -- Array, Strings, Linked List, Trees, ...
    display_order INT          NOT NULL,
    sent_date     DATE                    -- NULL = 미발송
);

CREATE INDEX idx_leetcode_unsent ON leetcode_problems (display_order)
    WHERE sent_date IS NULL;

INSERT INTO leetcode_problems (title, slug, difficulty, category, display_order) VALUES
-- Array
('Remove Duplicates from Sorted Array', 'remove-duplicates-from-sorted-array', 'Easy', 'Array', 1),
('Best Time to Buy and Sell Stock II', 'best-time-to-buy-and-sell-stock-ii', 'Easy', 'Array', 2),
('Rotate Array', 'rotate-array', 'Easy', 'Array', 3),
('Contains Duplicate', 'contains-duplicate', 'Easy', 'Array', 4),
('Single Number', 'single-number', 'Easy', 'Array', 5),
('Intersection of Two Arrays II', 'intersection-of-two-arrays-ii', 'Easy', 'Array', 6),
('Plus One', 'plus-one', 'Easy', 'Array', 7),
('Move Zeroes', 'move-zeroes', 'Easy', 'Array', 8),
('Two Sum', 'two-sum', 'Easy', 'Array', 9),
('Valid Sudoku', 'valid-sudoku', 'Medium', 'Array', 10),
('Rotate Image', 'rotate-image', 'Medium', 'Array', 11),
-- Strings
('Reverse String', 'reverse-string', 'Easy', 'Strings', 12),
('Reverse Integer', 'reverse-integer', 'Medium', 'Strings', 13),
('First Unique Character in a String', 'first-unique-character-in-a-string', 'Easy', 'Strings', 14),
('Valid Anagram', 'valid-anagram', 'Easy', 'Strings', 15),
('Valid Palindrome', 'valid-palindrome', 'Easy', 'Strings', 16),
('String to Integer (atoi)', 'string-to-integer-atoi', 'Medium', 'Strings', 17),
('Implement strStr()', 'find-the-index-of-the-first-occurrence-in-a-string', 'Easy', 'Strings', 18),
('Count and Say', 'count-and-say', 'Medium', 'Strings', 19),
('Longest Common Prefix', 'longest-common-prefix', 'Easy', 'Strings', 20),
('Longest Substring Without Repeating Characters', 'longest-substring-without-repeating-characters', 'Medium', 'Strings', 21),
-- Linked List
('Delete Node in a Linked List', 'delete-node-in-a-linked-list', 'Medium', 'Linked List', 22),
('Remove Nth Node From End of List', 'remove-nth-node-from-end-of-list', 'Medium', 'Linked List', 23),
('Reverse Linked List', 'reverse-linked-list', 'Easy', 'Linked List', 24),
('Merge Two Sorted Lists', 'merge-two-sorted-lists', 'Easy', 'Linked List', 25),
('Palindrome Linked List', 'palindrome-linked-list', 'Easy', 'Linked List', 26),
('Linked List Cycle', 'linked-list-cycle', 'Easy', 'Linked List', 27),
-- Trees
('Maximum Depth of Binary Tree', 'maximum-depth-of-binary-tree', 'Easy', 'Trees', 28),
('Validate Binary Search Tree', 'validate-binary-search-tree', 'Medium', 'Trees', 29),
('Symmetric Tree', 'symmetric-tree', 'Easy', 'Trees', 30),
('Binary Tree Level Order Traversal', 'binary-tree-level-order-traversal', 'Medium', 'Trees', 31),
('Convert Sorted Array to Binary Search Tree', 'convert-sorted-array-to-binary-search-tree', 'Easy', 'Trees', 32),
-- Sorting and Searching
('Merge Sorted Array', 'merge-sorted-array', 'Easy', 'Sorting', 33),
('First Bad Version', 'first-bad-version', 'Easy', 'Sorting', 34),
('Search in Rotated Sorted Array', 'search-in-rotated-sorted-array', 'Medium', 'Sorting', 35),
-- Dynamic Programming
('Climbing Stairs', 'climbing-stairs', 'Easy', 'Dynamic Programming', 36),
('Best Time to Buy and Sell Stock', 'best-time-to-buy-and-sell-stock', 'Easy', 'Dynamic Programming', 37),
('Maximum Subarray', 'maximum-subarray', 'Medium', 'Dynamic Programming', 38),
('House Robber', 'house-robber', 'Medium', 'Dynamic Programming', 39),
('Coin Change', 'coin-change', 'Medium', 'Dynamic Programming', 40),
('Jump Game', 'jump-game', 'Medium', 'Dynamic Programming', 41),
('Unique Paths', 'unique-paths', 'Medium', 'Dynamic Programming', 42),
-- Design
('Min Stack', 'min-stack', 'Medium', 'Design', 43),
('Shuffle an Array', 'shuffle-an-array', 'Medium', 'Design', 44),
-- Math
('Fizz Buzz', 'fizz-buzz', 'Easy', 'Math', 45),
('Count Primes', 'count-primes', 'Medium', 'Math', 46),
('Power of Three', 'power-of-three', 'Easy', 'Math', 47),
('Roman to Integer', 'roman-to-integer', 'Easy', 'Math', 48),
('Number of 1 Bits', 'number-of-1-bits', 'Easy', 'Math', 49),
('Hamming Distance', 'hamming-distance', 'Easy', 'Math', 50),
('Reverse Bits', 'reverse-bits', 'Easy', 'Math', 51),
('Pascal''s Triangle', 'pascals-triangle', 'Easy', 'Math', 52),
-- Others
('Valid Parentheses', 'valid-parentheses', 'Easy', 'Others', 53),
('Missing Number', 'missing-number', 'Easy', 'Others', 54),
('Majority Element', 'majority-element', 'Easy', 'Others', 55),
('Word Break', 'word-break', 'Medium', 'Others', 56),
('Intersection of Two Arrays', 'intersection-of-two-arrays', 'Easy', 'Others', 57);
