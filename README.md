first question answer in leetcode is
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        unordered_set<string> res;
        rec(s, 0, "", res);
        return res.size();
    }

private:
    void rec(const string& s, int i, string cur, unordered_set<string>& res) {
        if (cur.length() == 3) {
            if (cur[0] == cur[2]) {
                res.insert(cur);
            }
            return;
        }
        if (i == s.length()) {
            return;
        }
        rec(s, i + 1, cur, res);
        rec(s, i + 1, cur + s[i], res);
    }
};
