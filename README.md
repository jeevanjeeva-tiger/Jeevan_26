2381.shiffting letters ii :-
class Solution {
    public String shiftingLetters(String s, int[][] shifts) {
        int stringLength = s.length();
        // Difference array to hold the net shift values after performing all shift operations.
        int[] netShifts = new int[stringLength + 1];

        // Iterate over each shift operation and update the difference array accordingly.
        for (int[] shift : shifts) {
            int direction = (shift[2] == 0) ? -1 : 1;  // If the shift is left, make it negative.
            netShifts[shift[0]] += direction;          // Apply the shift to the start index.
            netShifts[shift[1] + 1] -= direction;      // Negate the shift after the end index.
        }

        // Apply the accumulated shifts to get the actual shift values.
        for (int i = 1; i <= stringLength; ++i) {
            netShifts[i] += netShifts[i - 1];
        }

        // Construct the result string after applying the shift to each character.
        StringBuilder resultStringBuilder = new StringBuilder();
        for (int i = 0; i < stringLength; ++i) {
            // Calculate the new character by shifting the current character accordingly.
            // The mod operation keeps the result within the range of the alphabet, 
            // and the addition of 26 before mod ensures the number is positive.
            int shiftedIndex = (s.charAt(i) - 'a' + netShifts[i] % 26 + 26) % 26;
            resultStringBuilder.append((char) ('a' + shiftedIndex));
        }
        // Convert the StringBuilder to a String and return the result.
        return resultStringBuilder.toString();
    }
}



1) two sum :-

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
    for (int i = 0; i < nums.size(); i++) {
        for (int j = i + 1; j < nums.size(); j++) {
            if (nums[i] + nums[j] == target) {
                return {i, j};
            }
        }
    }
    return {};
  }
};

2) add two numbers :-

class Solution {
 public:
  ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* curr = &dummy;
    int carry = 0;

    while (l1 != nullptr || l2 != nullptr || carry > 0) {
      if (l1 != nullptr) {
        carry += l1->val;
        l1 = l1->next;
      }
      if (l2 != nullptr) {
        carry += l2->val;
        l2 = l2->next;
      }
      curr->next = new ListNode(carry % 10);
      carry /= 10;
      curr = curr->next;
    }

    return dummy.next;
  }
};


3) longest  substring without repition of characters :-

import java.util.*;
class Solution 
{
    public int lengthOfLongestSubstring(String s)
    {
        int max=0;
        HashSet<Character>hash=new HashSet<Character>();
        int i=0;
        int j=0;
        while(i<s.length() && j<s.length())
        {
            if(hash.contains(s.charAt(j)))
            {
                hash.remove(s.charAt(i));
                i=i+1;
            }
            else
            {
             hash.add(s.charAt(j));
             j=j+1;
             max=Math.max(j-i,max);   
            }
        }
        return max;
    }
}
class Main{
  public static void main(String[] args){
    int answer = (new Solution()).lengthOfLongestSubstring("pwwkew");
    System.out.print(answer);
  }
}


4) median of two sorted arrays:-

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def f(i: int, j: int, k: int) -> int:
            if i >= m:
                return nums2[j + k - 1]
            if j >= n:
                return nums1[i + k - 1]
            if k == 1:
                return min(nums1[i], nums2[j])
            p = k // 2
            x = nums1[i + p - 1] if i + p - 1 < m else inf
            y = nums2[j + p - 1] if j + p - 1 < n else inf
            return f(i + p, j, k - p) if x < y else f(i, j + p, k - p)

        m, n = len(nums1), len(nums2)
        a = f(0, 0, (m + n + 1) // 2)
        b = f(0, 0, (m + n + 2) // 2)
        return (a + b) / 2

5) longest palondromic substring:-
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int n = s.length();
        String longest = "";
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (isPalindrome(s, i, j) && (j - i + 1) > longest.length()) {
                    longest = s.substring(i, j + 1);
                }
            }
        }
        return longest;
    }

    private boolean isPalindrome(String s, int start, int end) {
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}

6) zigzag conversion :-

  

class Solution {
public:
    string convert(string s, int numRows) {
        
        if (numRows == 1) return s;
        
        string res = "";
        int n = s.size();
        int cycleLen = 2 * numRows - 2;
        
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j + i < n; j += cycleLen) {
                res += s[j + i];
                if (i != 0 && i != numRows - 1 && j + cycleLen - i < n)
                    res += s[j + cycleLen - i];
            }
        }
        return res;
    }
};

7) reverse integer :-
    class Solution {
    public int reverse(int x) {

       boolean neg = false;      //check if negative number
        if(x<0)
        {
            neg = true;
            x = x*-1;
        }
        
        //Convert to string than reverse by Array
        String y = Integer.toString(x);
        
        StringBuilder sb=new StringBuilder(y);  
        y = sb.reverse().toString();  
        
        try {
            x = Integer.parseInt(y);
        } catch (NumberFormatException e) {
            return 0;
        }
        
        if(neg)
        {
            x*=-1;
        }
        return x;
    }
}

8) string to integer:-

class Solution {
  public int myAtoi(String str) {
    
    final int len = str.length();
    
    if (len == 0){
        return 0;
    }
    
    int index = 0;

    while (index < len && str.charAt(index) == ' '){
        ++index;
    }
    
    boolean isNegative = false;
    
    if (index < len) {
      
      if (str.charAt(index) == '-') {
        isNegative = true;
        ++index;
      } else if (str.charAt(index) == '+'){
          ++index;
      }
      
    }
    
    int result = 0;
    
    while (index < len && isDigit(str.charAt(index))) {
      
      int digit = str.charAt(index) - '0';
      
      if (result > (Integer.MAX_VALUE / 10) || (result == (Integer.MAX_VALUE / 10) && digit > 7)){
          return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
      }
      
      result = (result * 10) + digit;
      
      ++index;
    }
      
    return isNegative ? -result : result;
  }
  
  private boolean isDigit(char ch) {

    return ch >= '0' && ch <= '9';
  }
}


   9) palindeomic number:-
      class Solution {
    public boolean isPalindrome(int x) {
        int total = 0;
        int k = x;
        while(x > 0){
            
            int b = x%10;
            total = total*10 + b;
            x = x/10;
        
        }
        if(total == k)
            return true;
        return false;
    }
}

10) regular expression matchimg:-

 class Solution {
    public boolean isMatch(String s, String p) {
        // Initialize a 2D DP table
        // dp[i][j] will be true if s[0..i-1] matches p[0..j-1]
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        
        // Base case: Empty string matches with empty pattern
        dp[0][0] = true;
        
        // Handle patterns with '*' at the beginning
        for (int j = 1; j <= p.length(); j++) {
            if (p.charAt(j - 1) == '*') {
                dp[0][j] = dp[0][j - 2];
            }
        }
        
        // Fill the DP table
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= p.length(); j++) {
                if (p.charAt(j - 1) == s.charAt(i - 1) || p.charAt(j - 1) == '.') {
                    // Characters match, or pattern has '.', copy the diagonal value
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    // Check zero occurrence of the character before '*'
                    dp[i][j] = dp[i][j - 2];
                    // Check one or more occurrence of the character before '*'
                    if (p.charAt(j - 2) == s.charAt(i - 1) || p.charAt(j - 2) == '.') {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                }
            }
        }
        
        return dp[s.length()][p.length()];
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        // Test cases
        System.out.println(solution.isMatch("aa", "a"));     // Output: false
        System.out.println(solution.isMatch("aa", "a*"));    // Output: true
        System.out.println(solution.isMatch("ab", ".*"));    // Output: true
        System.out.println(solution.isMatch("aab", "c*a*b")); // Output: true
        System.out.println(solution.isMatch("mississippi", "mis*is*p*.")); // Output: false
    }
}


11) container with most water:-

/**
 * @author: Jiawei Wu
 * @create: 2020-03-15 10:34
 *
 * Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
 *
 * Note: You may not slant the container and n is at least 2.
 *
 *
 *
 * The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
 *
 *
 *
 * Example:
 *
 * Input: [1,8,6,2,5,4,8,3,7]
 * Output: 49
 **/

class Solution {
    public int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0;
        int right = height.length - 1;

        while (left < right) {
            int area = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, area);

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }

        return maxArea;
    }
}


12) integer to roman:-

/**
 * @author: Jiawei Wu
 * @create: 2020-03-15 10:34
 *
 * Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
 *
 * Note: You may not slant the container and n is at least 2.
 *
 *
 *
 * The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
 *
 *
 *
 * Example:
 *
 * Input: [1,8,6,2,5,4,8,3,7]
 * Output: 49
 **/

class Solution {
    public int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0;
        int right = height.length - 1;

        while (left < right) {
            int area = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, area);

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }

        return maxArea;
    }
}

13) ronman to integer:-

class Solution {
    public static int romanToInt(String s) {
        if (s == null || s.length() == 0)
            return -1;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        int len = s.length(), result = map.get(s.charAt(len - 1));
        for (int i = len - 2; i >= 0; i--) {
            if (map.get(s.charAt(i)) >= map.get(s.charAt(i + 1)))
                result += map.get(s.charAt(i));
            else
                result -= map.get(s.charAt(i));
        }
        return result;
    }
}

14) longest common prefix:-

class Solution {
    public String longestCommonPrefix(String[] strs) {
        int L = 0;
        for(int i=0;i<strs[0].length();i++){
            int ok = 1;
            for(int j=0;j<strs.length;j++){
                if(L>=strs[j].length() || strs[j].charAt(L)!=strs[0].charAt(L)){
                    ok = 0;
                    break;
                }
            }
            if(ok==1){
                L++;
            }
            else{
                break;
            }
        }
        return strs[0].substring(0,L);
    }
}

15) 3sum:-

    class Solution {
    public List<List<Integer>> threeSum(int[] arr) {
        Arrays.sort(arr);
        int n = arr.length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (arr[i] > 0) break; // Since arr[i] <= arr[l] <= arr[r], if a[i] > 0 then sum=arr[i]+arr[l]+arr[r] > 0
            int l = i + 1, r = n - 1;
            while (l < r) {
                int sum = arr[i] + arr[l] + arr[r];
                if (sum < 0) l++;
                else if (sum > 0) r--;
                else {
                    ans.add(Arrays.asList(arr[i], arr[l], arr[r]));
                    while (l+1 <= r && arr[l] == arr[l+1]) l++; // Skip duplicate nums[l]
                    l++;
                    r--;
                }
            }
            while (i+1 < n && arr[i+1] == arr[i]) i++; // Skip duplicate nums[i]
        }
        return ans;
    }
}

16) 3sum closest:-

class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int result=nums[0]+nums[1]+nums[nums.length-1];
        Arrays.sort(nums);
        for (int i=0;i<nums.length-2;i++) {
            int start=i+1,end=nums.length-1;
            while(start<end) {
                int sum=nums[i]+nums[start]+nums[end];
                if(sum>target) end--;
                else start++;
                if (Math.abs(sum-target)<Math.abs(result-target)) result=sum;
            }
        }
        return result;
    }
}

17) letter combination of phone letter:-

import java.util.ArrayList;
import java.util.List;

class Solution {
  
    // Let's declare a method to generate all possible letter combinations for a given phone number.
    public List<String> letterCombinations(String digits) {
        // A result list to store the final combinations.
        List<String> result = new ArrayList<>();
      
        // Return an empty list if the input digits string is empty.
        if (digits.length() == 0) {
            return result;
        }
      
        // Add an empty string as an initial value to start the combinations.
        result.add("");
      
        // Mapping of digit to letters as seen on a phone keypad.
        String[] digitToLetters = new String[] {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
      
        // Iterate over each digit in the input string.
        for (char digit : digits.toCharArray()) {
            // Get the corresponding letters for the current digit.
            String letters = digitToLetters[digit - '2'];
          
            // Temp list to hold the combinations for the current digit.
            List<String> temp = new ArrayList<>();
          
            // Combine each result in the list with each letter for the current digit.
            for (String combination : result) {
                for (char letter : letters.toCharArray()) {
                    // Add the new combination to the temp list.
                    temp.add(combination + letter);
                }
            }
          
            // Update the result list with the new set of combinations.
            result = temp;
        }
      
        // Return the complete list of combinations.
        return result;
    }
}

18) 4sum :-

    class Solution {
 public:
  vector<vector<int>> fourSum(vector<int>& nums, int target) {
    vector<vector<int>> ans;
    vector<int> path;
    ranges::sort(nums);
    nSum(nums, 4, target, 0, nums.size() - 1, path, ans);
    return ans;
  }

 private:
  // Finds n numbers that add up to the target in [l, r].
  void nSum(const vector<int>& nums, long n, long target, int l, int r,
            vector<int>& path, vector<vector<int>>& ans) {
    if (r - l + 1 < n || target < nums[l] * n || target > nums[r] * n)
      return;
    if (n == 2) {
      // Similar to the sub procedure in 15. 3Sum
      while (l < r) {
        const int sum = nums[l] + nums[r];
        if (sum == target) {
          path.push_back(nums[l]);
          path.push_back(nums[r]);
          ans.push_back(path);
          path.pop_back();
          path.pop_back();
          ++l;
          --r;
          while (l < r && nums[l] == nums[l - 1])
            ++l;
          while (l < r && nums[r] == nums[r + 1])
            --r;
        } else if (sum < target) {
          ++l;
        } else {
          --r;
        }
      }
      return;
    }

    for (int i = l; i <= r; ++i) {
      if (i > l && nums[i] == nums[i - 1])
        continue;
      path.push_back(nums[i]);
      nSum(nums, n - 1, target - nums[i], i + 1, r, path, ans);
      path.pop_back();
    }
  }
};

19) remove the nth node from end of the list:-

    class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *prev = nullptr,*curr = head,*start = nullptr;
        while(curr!=nullptr){
            n--;
            curr = curr->next;
            if(n<=0){
                if(start==nullptr){
                    start = head;
                }
                else{
                    prev = start;
                    start = start->next;
                }
            }
        }
        if(prev==nullptr){
            return start->next;
        }
        prev->next = start->next;
        return head;
    }
};

20) valid parenthisies:-

    class Solution {
    
    public boolean handleClosing(Stack<Character> s, char openBracket){
        if(s.size() == 0){
            return false;
        }else if(s.peek() != openBracket){
            return false;
        }else {
            s.pop();
        }
        return true;
    }
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(char ch : s.toCharArray()){
            if(ch == '(' || ch == '{' || ch == '['){
                stack.push(ch);
            }else if(ch == ')'){
                boolean val = handleClosing(stack,'(');
                if(val == false) return false;
            }else if(ch == '}'){
                boolean val = handleClosing(stack,'{');
                if(val == false) return false;
            }else if(ch == ']'){
                boolean val = handleClosing(stack,'[');
                if(val == false) return false;
            }
        }
        
        if(stack.size() > 0)return false;
        return true;
    }
}

21) merge two sorted lists:-

    /**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if(list1 == null){ return list2;}
        if(list2 == null){ return list1;}
 
        ListNode resNode = new ListNode();
        
        //init
        if(list1.val>list2.val)
        {
            resNode.val = list2.val;
            list2 = list2.next;
        }
        else
        {
            resNode.val = list1.val;
            list1 = list1.next;
        }
        
        ListNode lastNode = resNode;
        
        //iterate to one listNode next is null
        while(list1!=null && list2!=null)
        {
            ListNode nowNode = new ListNode();
            if(list1.val>list2.val)
            {
                nowNode.val = list2.val;
                list2 = list2.next;
            }
            else
            {
                nowNode.val = list1.val;
                list1 = list1.next;
            }
            
            lastNode.next = nowNode;
            lastNode = nowNode;
        }
        
        //one linked list will have some remain node
        if(list1!=null)
        {
            lastNode.next = list1;
        }
        else
        {
            lastNode.next = list2;
        }
    
        return resNode;
    }
}

22) generate parenthiesis:-

    class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<String>();
        backtrack(list, "", 0, 0, n);
        return list;
    }
    public void backtrack(List<String> list, String str, int open, int close, int max){
        if(str.length() == max*2){
            list.add(str);
            return;
        }        
        if(open < max){
            backtrack(list, str+"(", open+1, close, max);
        }
        if(close < open){
            backtrack(list, str+")", open, close+1, max);
        }
    }
}

23) merge k sorted lists:-

    class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists.length==0){
            return null;
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(lists.length, (a,b)-> a.val-b.val);
        for(ListNode node:lists){
            if(node!=null){
                pq.add(node);
            }
        }
        ListNode dummy = new ListNode(0);
        ListNode head = dummy;
        while(!pq.isEmpty()){
            ListNode node = pq.poll();
            if(node.next!=null){
                pq.add(node.next);
            }
            dummy.next = node;
            dummy = node;
        }
        return head.next;
    }
}

24) swap node in pairs:-

    class Solution {
  public ListNode swapPairs(ListNode head) {
    final int length = getLength(head);
    ListNode dummy = new ListNode(0, head);
    ListNode prev = dummy;
    ListNode curr = head;

    for (int i = 0; i < length / 2; ++i) {
      ListNode next = curr.next;
      curr.next = next.next;
      next.next = curr;
      prev.next = next;
      prev = curr;
      curr = curr.next;
    }

    return dummy.next;
  }

  private int getLength(ListNode head) {
    int length = 0;
    for (ListNode curr = head; curr != null; curr = curr.next)
      ++length;
    return length;
  }
}
// code provided by PROGIEZ

25) reverse nodes in k group:-

    class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        // A dummy node with 0 as value and pointing to the head of the list
        ListNode dummyNode = new ListNode(0, head);
        ListNode predecessor = dummyNode, current = dummyNode;

        // Iterate through the list
        while (current.next != null) {
            // Check if there are k nodes to reverse
            for (int i = 0; i < k && current != null; i++) {
                current = current.next;
            }
            // If less than k nodes remain, no more reversing is needed
            if (current == null) {
                return dummyNode.next;
            }

            // Temporarily store the next segment to be addressed after reversal
            ListNode temp = current.next;
            // Detach the k nodes from the rest of the list
            current.next = null;
            // 'start' will be the new tail after the reversal
            ListNode start = predecessor.next;
            // Reverse the k nodes
            predecessor.next = reverseList(start);
            // Connect the new tail to the temp segment stored before
            start.next = temp;
            // Move the predecessor and current pointers k nodes ahead
            predecessor = start;
            current = predecessor;
        }
        return dummyNode.next;
    }

    /**
     * Helper method to reverse the linked list.
     * 
     * @param head The head of the list to be reversed.
     * @return The new head of the reversed list.
     */
    private ListNode reverseList(ListNode head) {
        ListNode previous = null, currentNode = head;

        // Traverse the list and reverse the links
        while (currentNode != null) {
            ListNode nextNode = currentNode.next;
            currentNode.next = previous;
            previous = currentNode;
            currentNode = nextNode;
        }
        // Return the new head of the reversed list
        return previous;
    }
}

26) 
